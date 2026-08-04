/**
 * multiview-stitcher in the browser.
 *
 * The page itself does no image work. It wires together four things and then
 * stays out of the way:
 *
 *   - a service worker that turns HTTP requests into local file reads, Python
 *     chunk computations and pool dispatch;
 *   - an fs worker that owns the directory handles the user granted;
 *   - one persistent session worker holding the dataset, plus a pool of
 *     compute workers;
 *   - Neuroglancer in an iframe, driven by viewer state that Python builds.
 *
 * Keeping the main thread free matters: Python workers block on synchronous
 * requests that this thread has to keep routing.
 */

const APP_BASE = new URL(".", window.location.href).pathname;
const API_BASE = `${APP_BASE}__mvs__`;

const state = {
  config: null,
  session: null, // most recent `describe()` result
  sessionSpec: null, // snapshot compute workers rebuild from
  previewRoute: null,
  transformKey: null,
  mounts: [],
};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

const $ = (selector) => document.querySelector(selector);

function log(message, level = "info") {
  const line = document.createElement("div");
  line.className = `log-line log-${level}`;
  line.textContent = `${new Date().toLocaleTimeString()}  ${message}`;
  const box = $("#log");
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}

function setStatus(message, busy = false) {
  $("#status").textContent = message;
  $("#status").classList.toggle("busy", busy);
}

function setBusy(busy) {
  for (const button of document.querySelectorAll("button[data-action]")) {
    button.disabled = busy || !state.session;
  }
  $("#worker-count").disabled = busy;
}

/** Promise-based request/response over a Worker's postMessage. */
class WorkerChannel {
  constructor(url, name) {
    this.worker = new Worker(url);
    this.name = name;
    this.pending = new Map();
    this.nextId = 1;
    this.busy = false;

    this.worker.onmessage = (event) => {
      const { id, type } = event.data;
      if (type === "log") {
        log(`${name}: ${event.data.message}`);
        return;
      }
      const entry = this.pending.get(id);
      if (!entry) return;
      this.pending.delete(id);
      if (event.data.error) entry.reject(new Error(event.data.error));
      else entry.resolve(event.data);
    };

    this.worker.onerror = (event) => {
      log(`${name}: ${event.message}`, "error");
    };
  }

  send(message, transfer = []) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, ...message }, transfer);
    });
  }
}

// ---------------------------------------------------------------------------
// Worker pool
// ---------------------------------------------------------------------------

class ComputePool {
  constructor() {
    this.workers = [];
    this.queue = [];
  }

  get size() {
    return this.workers.length;
  }

  /** Grow or shrink the pool to `count` booted workers. */
  async resize(count, config) {
    while (this.workers.length > count) {
      this.workers.pop().worker.terminate();
    }

    const booting = [];
    while (this.workers.length < count) {
      const index = this.workers.length;
      const channel = new WorkerChannel("compute-worker.js", `worker ${index}`);
      this.workers.push(channel);
      booting.push(channel.send({ type: "boot", config, name: `worker ${index}` }));
    }

    if (booting.length) {
      log(`starting ${booting.length} compute worker(s)`);
      await Promise.all(booting);
    }
  }

  /** Run one job on the next free worker, queueing when all are busy. */
  run(job) {
    return new Promise((resolve, reject) => {
      this.queue.push({ job, resolve, reject });
      this.pump();
    });
  }

  pump() {
    if (!this.queue.length) return;

    const worker = this.workers.find((candidate) => !candidate.busy);
    if (!worker) return;

    const entry = this.queue.shift();
    worker.busy = true;

    worker
      .send(entry.job.message, entry.job.transfer || [])
      .then(entry.resolve, entry.reject)
      .finally(() => {
        worker.busy = false;
        this.pump();
      });
  }

  /** Run every task, keeping all workers busy; results keep the input order. */
  async dispatch(tasks, sessionSpec) {
    return await Promise.all(
      tasks.map(async (task) => {
        const response = await this.run({
          message: { type: "task", task: { ...task, session: task.session || sessionSpec } },
        });
        return response.result;
      }),
    );
  }
}

const pool = new ComputePool();
let fsWorker = null;
let sessionWorker = null;

// ---------------------------------------------------------------------------
// Service worker routing
// ---------------------------------------------------------------------------

/**
 * Answer the service worker.
 *
 * Chunk requests go to a free compute worker when the pool is up (so that
 * lazily fused previews render in parallel) and fall back to the session
 * worker, which always has the live session.
 */
async function handleServiceWorkerMessage(event) {
  const { type, port } = event.data;
  if (!port) return;

  const reply = (payload, transfer = []) => port.postMessage(payload, transfer);

  try {
    if (type === "fs.read") {
      const response = await fsRequest({
        type: "read",
        mount: event.data.mount,
        path: event.data.path,
      });
      reply(response, response.found ? [response.data] : []);
      return;
    }

    if (type === "zarr.read") {
      const message = { type: "serve", route: event.data.route, key: event.data.key };
      let response;

      if (pool.size && state.sessionSpec) {
        response = await pool.run({
          message: { ...message, session: state.sessionSpec },
        });
      } else {
        response = await sessionWorker.send(message);
      }

      reply(
        {
          found: response.found,
          data: response.data,
          contentType: response.contentType,
        },
        response.found ? [response.data] : [],
      );
      return;
    }

    if (type === "rpc") {
      const payload = event.data.payload || {};
      if (event.data.endpoint !== "dispatch") {
        throw new Error(`unknown rpc endpoint '${event.data.endpoint}'`);
      }
      const results = await pool.dispatch(payload.tasks || [], state.sessionSpec);
      reply({ result: { results } });
      return;
    }

    throw new Error(`unknown service-worker message '${type}'`);
  } catch (error) {
    reply({ error: String((error && error.message) || error) });
  }
}

function fsRequest(message) {
  return new Promise((resolve, reject) => {
    const channel = new MessageChannel();
    channel.port1.onmessage = (event) => {
      if (event.data && event.data.error) reject(new Error(event.data.error));
      else resolve(event.data);
    };
    fsWorker.worker.postMessage({ ...message, port: channel.port2 }, [channel.port2]);
  });
}

// ---------------------------------------------------------------------------
// Neuroglancer
// ---------------------------------------------------------------------------

function showViewerState(ngState) {
  const frame = $("#viewer");
  const hash = "#!" + encodeURIComponent(JSON.stringify(ngState));

  if (frame.dataset.loaded === "true") {
    try {
      // Updating the hash keeps the camera; reloading the iframe would not.
      frame.contentWindow.location.hash = hash;
      return;
    } catch (error) {
      log(`could not update the viewer in place: ${error.message}`, "warn");
    }
  }

  frame.src = `neuroglancer/index.html${hash}`;
  frame.dataset.loaded = "true";
}

async function refreshViewer() {
  if (!state.session) return;

  const ngState = await command("neuroglancer_state", {
    transform_key: state.transformKey,
    base_url: window.location.origin,
    preview_route: state.previewRoute,
  });

  showViewerState(ngState);
}

// ---------------------------------------------------------------------------
// Session commands
// ---------------------------------------------------------------------------

async function command(name, payload) {
  const response = await sessionWorker.send({
    type: "command",
    command: name,
    payload,
  });
  return response.result;
}

async function refreshSessionSpec() {
  state.sessionSpec = await command("spec", {});
}

function renderTransformKeys(keys) {
  const select = $("#transform-key");
  select.innerHTML = "";

  for (const key of keys) {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = key;
    select.appendChild(option);
  }

  if (!keys.includes(state.transformKey)) {
    state.transformKey = keys[keys.length - 1] || null;
  }
  select.value = state.transformKey || "";
  select.disabled = keys.length < 2;
}

function renderViews(described) {
  const list = $("#views");
  list.innerHTML = "";

  for (const view of described.views) {
    const level = view.levels[0];
    const shape = Object.entries(level.shape)
      .map(([dim, size]) => `${dim}:${size}`)
      .join(" ");
    const item = document.createElement("li");
    item.innerHTML = `<strong>${view.name}</strong><span>${shape} · ${view.dtype} · ${view.levels.length} level(s)</span>`;
    list.appendChild(item);
  }

  $("#dataset-summary").textContent =
    `${described.n_views} view(s), ${described.views[0].ndim}D`;
}

async function applyDescribed(described) {
  state.session = described;
  state.previewRoute = null;
  renderViews(described);
  renderTransformKeys(described.transform_keys);
  await refreshSessionSpec();
  await refreshViewer();
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

async function loadDirectory(handle) {
  const mount = crypto.randomUUID().slice(0, 8);
  await fsRequest({ type: "mount", mount, handle });
  state.mounts.push(mount);

  const { images } = await fsRequest({ type: "discover", mount });
  if (!images.length) {
    throw new Error(
      "no OME-Zarr found - drop either an OME-Zarr directory or a folder containing several of them",
    );
  }

  const sources = images.map((image) => ({
    url: `${API_BASE}/fs/${mount}${image.path ? "/" + image.path : ""}`,
    name: image.name,
  }));

  log(`found ${sources.length} OME-Zarr image(s)`);
  setStatus("opening images", true);

  const described = await command("load", { sources });
  await applyDescribed(described);

  setStatus(`${described.n_views} view(s) loaded`);
  setBusy(false);
}

async function doRegister() {
  setBusy(true);
  setStatus("registering", true);
  const started = performance.now();

  try {
    const result = await command("register", {
      options: { new_transform_key: "registered" },
      distribute: pool.size > 0,
    });

    log(
      `registered ${result.params.length} view(s) in ` +
        `${((performance.now() - started) / 1000).toFixed(1)}s` +
        (pool.size ? ` on ${pool.size} worker(s)` : ""),
    );

    state.session.transform_keys = result.transform_keys;
    state.transformKey = result.transform_key;
    state.previewRoute = null;
    renderTransformKeys(result.transform_keys);
    await refreshSessionSpec();
    await refreshViewer();
    setStatus("registered");
  } finally {
    setBusy(false);
  }
}

async function doFusePreview() {
  setBusy(true);
  setStatus("preparing the fused preview", true);

  try {
    const result = await command("fuse_preview", {
      options: { transform_key: state.transformKey },
    });
    state.previewRoute = result.route;
    await refreshSessionSpec();
    await refreshViewer();

    const shape = Object.entries(result.metadata.levels[0].shape)
      .map(([dim, size]) => `${dim}:${size}`)
      .join(" ");
    log(`fused preview ready (${shape}); chunks are computed on demand`);
    setStatus("fused preview added to the viewer");
  } finally {
    setBusy(false);
  }
}

async function doFuseToDisk() {
  const handle = await window.showDirectoryPicker({ mode: "readwrite" });

  setBusy(true);
  setStatus("fusing to OME-Zarr", true);

  try {
    // Writing needs a real filesystem, so the output directory is mounted into
    // the session worker's Emscripten filesystem and flushed when done.
    await sessionWorker.send({ type: "mount_output", handle });
    const result = await command("fuse_to_zarr", {
      options: {
        transform_key: state.transformKey,
        output_zarr_url: "/output/fused.ome.zarr",
      },
    });
    await sessionWorker.send({ type: "sync_output" });

    log(`wrote ${result.n_blocks} block(s) to fused.ome.zarr`);
    setStatus("fused image written to disk");
  } finally {
    setBusy(false);
  }
}

// ---------------------------------------------------------------------------
// Start-up
// ---------------------------------------------------------------------------

async function boot() {
  state.config = await (await fetch("config.json")).json();

  const manifest = await (await fetch("packages/manifest.json")).json();
  const config = {
    ...state.config,
    wheel_url: new URL(`packages/${manifest.wheel}`, window.location.href).href,
    api_base: API_BASE,
  };
  state.runtimeConfig = config;

  if (!("serviceWorker" in navigator)) {
    throw new Error("this browser has no service worker support");
  }

  log("registering the service worker");
  const registration = await navigator.serviceWorker.register("sw.js");
  await navigator.serviceWorker.ready;
  if (!navigator.serviceWorker.controller) {
    // First visit: the worker activated after this page loaded, so nothing is
    // routed through it yet.
    log("reloading so the service worker can serve local files");
    window.location.reload();
    return;
  }
  registration.update().catch(() => {});
  navigator.serviceWorker.addEventListener("message", handleServiceWorkerMessage);

  fsWorker = new WorkerChannel("fs-worker.js", "fs");

  sessionWorker = new WorkerChannel("session-worker.js", "session");
  setStatus("starting the Python runtime", true);
  const info = (await sessionWorker.send({ type: "boot", config })).result;
  log(
    `python ${info.python} · numpy ${info.numpy} · zarr ${info.zarr} · ` +
      `dask ${info.dask} · multiview-stitcher ${info.multiview_stitcher}`,
  );

  const select = $("#worker-count");
  select.innerHTML = "";
  for (let count = 0; count <= state.config.max_n_workers; count += 1) {
    const option = document.createElement("option");
    option.value = String(count);
    option.textContent = count === 0 ? "none (session worker only)" : String(count);
    select.appendChild(option);
  }
  const suggested = Math.min(
    state.config.max_n_workers,
    Math.max(1, (navigator.hardwareConcurrency || 4) - 1),
  );
  select.value = String(Math.min(suggested, state.config.default_n_workers));

  setStatus("drop a folder to begin");
  $("#dropzone").classList.remove("disabled");
}

function wireUi() {
  const dropzone = $("#dropzone");

  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));

  dropzone.addEventListener("drop", async (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");

    const item = event.dataTransfer.items[0];
    if (!item || !item.getAsFileSystemHandle) {
      log("this browser cannot read dropped folders; use the browse button", "error");
      return;
    }

    try {
      const handle = await item.getAsFileSystemHandle();
      if (handle.kind !== "directory") {
        log("drop a folder, not a single file", "error");
        return;
      }
      await withPool(() => loadDirectory(handle));
    } catch (error) {
      log(error.message, "error");
      setStatus("failed to open the dropped folder");
    }
  });

  $("#browse").addEventListener("click", async () => {
    try {
      const handle = await window.showDirectoryPicker({ mode: "read" });
      await withPool(() => loadDirectory(handle));
    } catch (error) {
      if (error.name !== "AbortError") log(error.message, "error");
    }
  });

  $("#transform-key").addEventListener("change", async (event) => {
    state.transformKey = event.target.value;
    log(`showing transform key '${state.transformKey}'`);
    await refreshViewer();
  });

  for (const [action, handler] of Object.entries({
    register: doRegister,
    "fuse-preview": doFusePreview,
    "fuse-disk": doFuseToDisk,
  })) {
    $(`button[data-action="${action}"]`).addEventListener("click", async () => {
      try {
        await withPool(handler);
      } catch (error) {
        log(error.message, "error");
        setStatus("failed");
        setBusy(false);
      }
    });
  }
}

/** Make sure the compute pool matches the current selection, then act. */
async function withPool(action) {
  const requested = Number($("#worker-count").value);
  if (requested !== pool.size) {
    setStatus("starting compute workers", true);
    await pool.resize(requested, state.runtimeConfig);
  }
  return await action();
}

wireUi();
boot().catch((error) => {
  log(error.message, "error");
  setStatus("failed to start");
});
