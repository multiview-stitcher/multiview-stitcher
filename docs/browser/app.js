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

/* global mvsRoutes */

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

  // Mirrored so that everything is in one place: the viewer reports its own
  // failures to the console, and correlating the two matters more than
  // keeping the console clean.
  if (level === "error") console.error("[multiview-stitcher]", message);
  else if (level === "warn") console.warn("[multiview-stitcher]", message);
}

const reportedFailures = new Map();

/** Log a serving failure once per distinct message, with a repeat count. */
function logServingFailure(type, message) {
  const key = `${type}: ${message}`;
  const seen = (reportedFailures.get(key) || 0) + 1;
  reportedFailures.set(key, seen);

  if (seen === 1) {
    log(key, "error");
  } else if (seen === 5 || seen % 50 === 0) {
    log(`${key} (${seen}x)`, "error");
  }
}


function setStatus(message, busy = false) {
  $("#status").textContent = message;
  $("#status").classList.toggle("busy", busy);
}

function hasViews() {
  return Boolean(state.session && state.session.n_views);
}

function setBusy(busy) {
  for (const button of document.querySelectorAll("button[data-action]")) {
    // Loading actions stay available while there is no data; the processing
    // ones need views first.
    const needsData = button.dataset.needsData !== "false";
    button.disabled = busy || (needsData && !hasViews());
  }
  for (const button of document.querySelectorAll("button[data-load]")) {
    button.disabled = busy;
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
      const channel = new WorkerChannel(
        `compute-worker.js?v=${config.build || "dev"}`,
        `worker ${index}`,
      );
      // Held busy until its Python runtime is up: a worker that is dispatched
      // to while still booting answers "the Python runtime is still starting",
      // which surfaces as a chunk that silently fails to render.
      channel.busy = true;
      this.workers.push(channel);
      booting.push(
        channel
          .send({ type: "boot", config, name: `worker ${index}` })
          .then(() => {
            channel.busy = false;
            this.pump();
          }),
      );
    }

    if (booting.length) {
      log(`starting ${booting.length} compute worker(s)`);
      await Promise.all(booting);
    }
  }

  /**
   * Run one job on the next free worker, queueing when all are busy.
   *
   * `timeoutMs` bounds how long the caller waits, not the worker: a worker
   * blocked inside Python cannot be interrupted, so the caller is released to
   * try elsewhere while the worker is left to finish and free itself.
   */
  run(job, { timeoutMs } = {}) {
    return new Promise((resolve, reject) => {
      this.queue.push({ job, resolve, reject, timeoutMs });
      this.pump();
    });
  }

  pump() {
    if (!this.queue.length) return;

    const worker = this.workers.find((candidate) => !candidate.busy);
    if (!worker) return;

    const entry = this.queue.shift();
    worker.busy = true;

    const sent = worker.send(entry.job.message, entry.job.transfer || []);

    let guarded = sent;
    if (entry.timeoutMs) {
      guarded = Promise.race([
        sent,
        new Promise((_, reject) =>
          setTimeout(
            () =>
              reject(
                new Error(`${worker.name} did not answer in ${entry.timeoutMs} ms`),
              ),
            entry.timeoutMs,
          ),
        ),
      ]);
    }

    guarded.then(entry.resolve, entry.reject);
    sent.finally(() => {
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

// ---------------------------------------------------------------------------
// One active tab
// ---------------------------------------------------------------------------

// A single service worker serves every open tab, and it answers with whichever
// tab replies first. Each tab, though, owns its own directory handles, its own
// session and its own Python workers - so a second tab can answer for the
// first and there is no way for either to tell from the request alone. The
// newest tab takes ownership and the others stand down.
const TAB_ID = crypto.randomUUID();
const tabChannel =
  typeof BroadcastChannel === "undefined"
    ? null
    : new BroadcastChannel("multiview-stitcher");
let tabIsActive = true;

function standDown() {
  if (!tabIsActive) return;
  tabIsActive = false;
  $("#inactive").hidden = false;
  log("another tab took over; this one is now inactive", "warn");
}

function claimTab() {
  if (!tabChannel) return;

  tabChannel.onmessage = (event) => {
    if (event.data && event.data.type === "claim" && event.data.id !== TAB_ID) {
      standDown();
    }
  };
  tabChannel.postMessage({ type: "claim", id: TAB_ID });
}

const pool = new ComputePool();
let fsWorker = null;
let sessionWorker = null;

// How long to wait for a compute worker to answer a chunk request before
// serving it from the session worker instead.
const POOL_SERVE_TIMEOUT_MS = 90 * 1000;
// After this many failures the pool stops being asked for chunks at all: a
// viewer that renders slowly beats one that renders nothing.
const POOL_SERVE_GIVE_UP_AFTER = 3;
let poolServeFailures = 0;


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

  // Decline anything this tab does not own, so the request is offered to the
  // tab that does.
  if (!tabIsActive) {
    reply({ notMine: true });
    return;
  }
  if (!fsWorker || !sessionWorker) {
    reply({ notMine: true });
    return;
  }
  if (
    (type === "fs.read" || type === "fs.write") &&
    !state.mounts.includes(event.data.mount)
  ) {
    reply({ notMine: true });
    return;
  }
  const sessionId = state.session && state.session.session_id;

  if (
    type === "zarr.read" &&
    !mvsRoutes.ownsRoute(sessionId, event.data.route)
  ) {
    reply({ notMine: true });
    return;
  }
  if (type === "rpc" && !sessionId) {
    reply({ notMine: true });
    return;
  }

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

    if (type === "fs.write") {
      const data = event.data.data;
      await fsRequest(
        data === null
          ? { type: "remove", mount: event.data.mount, path: event.data.path }
          : {
              type: "write",
              mount: event.data.mount,
              path: event.data.path,
              data,
            },
        data === null ? [] : [data],
      );
      reply({ ok: true });
      return;
    }

    if (type === "zarr.read") {
      const message = { type: "serve", route: event.data.route, key: event.data.key };
      let response;

      // Metadata always comes from the session worker, which owns the live
      // session and cannot be wrong about it. Only chunk bytes - the
      // expensive part - go to the pool, where a worker has to reconstruct
      // the image from a spec first. A layer whose metadata fails to load
      // shows up as an empty source with nothing to render.
      const isMetadata = mvsRoutes.isMetadataKey(event.data.key);
      const usePool =
        !isMetadata &&
        pool.size &&
        state.sessionSpec &&
        state.sessionSpec.session_id &&
        poolServeFailures < POOL_SERVE_GIVE_UP_AFTER;

      if (usePool) {
        try {
          response = await pool.run(
            { message: { ...message, session: state.sessionSpec } },
            { timeoutMs: POOL_SERVE_TIMEOUT_MS },
          );
        } catch (error) {
          // A compute worker rebuilds the session from the spec, which can
          // fail for reasons the session worker is immune to - it already
          // holds the opened data. Falling back keeps the viewer working, and
          // the log names the side that failed.
          poolServeFailures += 1;
          logServingFailure("zarr.read on a compute worker", error.message);
          if (poolServeFailures === POOL_SERVE_GIVE_UP_AFTER) {
            log(
              "serving chunks from the session worker only from now on",
              "warn",
            );
          }
          response = await sessionWorker.send(message);
        }
      } else {
        response = await sessionWorker.send(message);
      }

      if (!response.found && response.reason) {
        // Probing for keys that do not exist is normal; a route the app is
        // currently showing coming back empty is not.
        // Neuroglancer probes each source for both zarr formats, so a 404
        // for `zarr.json` or a root `.zarray` on a group is the expected
        // answer, not a failure.
        const isFormatProbe =
          event.data.key === "zarr.json" || event.data.key === ".zarray";
        const isCurrent =
          event.data.route === state.previewRoute ||
          event.data.route.includes("/view_");
        if (isCurrent && !isFormatProbe) {
          logServingFailure(
            `empty ${event.data.key} for ${event.data.route}`,
            response.reason,
          );
        }
      }

      reply(
        {
          found: response.found,
          data: response.data,
          contentType: response.contentType,
          reason: response.reason,
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
    const message = String((error && error.message) || error);
    // Requests the viewer makes are invisible unless we say something: a
    // failed chunk otherwise just renders as empty space.
    logServingFailure(type, message);
    reply({ error: message });
  }
}


function fsRequest(message, transfer = []) {
  return new Promise((resolve, reject) => {
    const channel = new MessageChannel();
    channel.port1.onmessage = (event) => {
      if (event.data && event.data.error) reject(new Error(event.data.error));
      else resolve(event.data);
    };
    fsWorker.worker.postMessage({ ...message, port: channel.port2 }, [
      channel.port2,
      ...transfer,
    ]);
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
  if (!hasViews()) return;

  const ngState = await command("neuroglancer_state", {
    transform_key: state.transformKey,
    base_url: window.location.origin,
    // The service worker only claims URLs inside its own scope, so Python
    // must build viewer URLs below this prefix rather than at the site root.
    api_base: API_BASE,
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
  const spec = await command("spec", {});

  // Only a spec a worker can actually rebuild from is worth dispatching. An
  // unusable one would make every compute worker raise, and the request would
  // land back on the session worker anyway - so keep the fallback and say so
  // rather than failing once per chunk.
  if (!spec || !spec.session_id || !(spec.sources || []).length) {
    state.sessionSpec = null;
    log(
      `session spec is not usable by compute workers: ${JSON.stringify(spec)}`,
      "warn",
    );
    return;
  }

  state.sessionSpec = spec;
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

  described.views.forEach((view, index) => {
    const level = view.levels[0];
    const shape = Object.entries(level.shape)
      .map(([dim, size]) => `${dim}:${size}`)
      .join(" ");

    const item = document.createElement("li");

    const text = document.createElement("div");
    const name = document.createElement("strong");
    // The viewer names its layers the same way, so the two lists line up.
    name.textContent = `${index}: ${view.name}`;
    const detail = document.createElement("span");
    detail.textContent = `${shape} · ${view.dtype} · ${view.levels.length} level(s)`;
    text.append(name, detail);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "remove";
    remove.title = `Remove ${view.name}`;
    remove.textContent = "×";
    remove.addEventListener("click", async () => {
      try {
        await removeView(index, view.name);
      } catch (error) {
        log(error.message, "error");
        setBusy(false);
      }
    });

    item.append(text, remove);
    list.appendChild(item);
  });

  $("#dataset-summary").textContent = described.n_views
    ? `${described.n_views} view(s), ${described.views[0].ndim}D`
    : "no data loaded";
}

async function applyDescribed(described) {
  poolServeFailures = 0;
  reportedFailures.clear();
  state.session = described;
  state.previewRoute = null;
  renderViews(described);
  renderTransformKeys(described.transform_keys);
  await refreshSessionSpec();
  await refreshViewer();
  setBusy(false);
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

async function loadDirectory(handle) {
  // The fs worker returns the existing mount when this folder is already
  // known, so re-dropping it addresses the same URLs instead of duplicating.
  const response = await fsRequest({
    type: "mount",
    mount: crypto.randomUUID().slice(0, 8),
    handle,
  });
  const mount = response.mount;
  if (!state.mounts.includes(mount)) state.mounts.push(mount);

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

  // Dropping more data adds tiles to the session instead of replacing it, so
  // a dataset can be assembled from several folders. "Clear" starts over.
  const append = Boolean(state.session && state.session.n_views);
  log(
    `found ${sources.length} OME-Zarr image(s); ` +
      (append ? "adding to the loaded views" : "opening"),
  );
  setStatus("opening images", true);

  const described = await command("load", { sources, replace: !append });
  await applyDescribed(described);

  setStatus(`${described.n_views} view(s) loaded`);
  setBusy(false);
}

async function loadExample(name) {
  const append = Boolean(state.session && state.session.n_views);
  setStatus("generating the example dataset", true);

  const described = await command("load_example", { name, replace: !append });
  await applyDescribed(described);

  log(`loaded the '${name}' example: ${described.n_views} view(s)`);
  setStatus(`${described.n_views} view(s) loaded`);
  setBusy(false);
}

async function removeView(index, name) {
  setStatus("removing view", true);
  const described = await command("remove", { index });
  state.previewRoute = null;
  await applyDescribed(described);
  log(`removed '${name}'`);
  setStatus(
    described.n_views ? `${described.n_views} view(s) loaded` : "drop a folder to begin",
  );
}

async function clearSession() {
  const described = await command("clear", {});
  state.previewRoute = null;
  await applyDescribed(described);
  log("cleared all views");
  setStatus("drop a folder to begin");
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

const OUTPUT_NAME = "fused.ome.zarr";

async function doFuseToDisk() {
  const handle = await window.showDirectoryPicker({ mode: "readwrite" });

  setBusy(true);
  setStatus("preparing the output", true);

  try {
    // The output directory is mounted like any other, and written through the
    // same service worker that serves reads: one HTTP request per chunk file.
    // Every worker shares this one handle, and because each writes a distinct
    // file they can all write at once, with no flush step to coordinate.
    const mounted = await fsRequest({
      type: "mount",
      mount: crypto.randomUUID().slice(0, 8),
      handle,
    });
    const mount = mounted.mount;
    if (!state.mounts.includes(mount)) state.mounts.push(mount);

    // Clear any previous output first: an HTTP-backed zarr store cannot list
    // its contents, so it cannot replace an existing array by itself.
    await fsRequest({ type: "remove", mount, path: OUTPUT_NAME });

    const started = performance.now();
    setStatus("fusing to OME-Zarr", true);

    const result = await command("fuse_to_zarr", {
      options: {
        transform_key: state.transformKey,
        output_zarr_url: `${API_BASE}/fs/${mount}/${OUTPUT_NAME}`,
      },
      distribute: pool.size > 0,
      n_workers: pool.size || 1,
    });

    log(
      `wrote ${result.n_blocks} block(s) across ${result.levels.length} ` +
        `resolution level(s) to ${OUTPUT_NAME} in ` +
        `${((performance.now() - started) / 1000).toFixed(1)}s` +
        (pool.size ? ` on ${pool.size} worker(s)` : ""),
    );
    setStatus("fused image written to disk");
  } finally {
    setBusy(false);
  }
}

// ---------------------------------------------------------------------------
// Start-up
// ---------------------------------------------------------------------------

async function boot() {
  // Never from cache: these describe which build to load, so a stale copy
  // would pin the whole app to an old one.
  const noStore = { cache: "no-store" };
  state.config = await (await fetch("config.json", noStore)).json();
  // The page bootstrap already read this to version its own scripts.
  const manifest =
    window.__mvsManifest ||
    (await (await fetch("packages/manifest.json", noStore)).json());

  // A rebuild of the same commit produces a wheel with the same filename, and
  // micropip fetches it from inside a worker - where a page reload does not
  // bypass the HTTP cache. Without this the runtime silently keeps running
  // yesterday's Python behind today's JavaScript. The same applies to the
  // worker scripts, which `new Worker`/`importScripts` also load from cache.
  const build = String(manifest.sha256 || "dev").slice(0, 12);
  state.build = build;

  const config = {
    ...state.config,
    wheel_url: new URL(
      `packages/${manifest.wheel}?v=${build}`,
      window.location.href,
    ).href,
    api_base: API_BASE,
    build,
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

  fsWorker = new WorkerChannel(`fs-worker.js?v=${build}`, "fs");

  sessionWorker = new WorkerChannel(`session-worker.js?v=${build}`, "session");
  setStatus("starting the Python runtime", true);
  const info = (await sessionWorker.send({ type: "boot", config })).result;
  log(
    `python ${info.python} · numpy ${info.numpy} · zarr ${info.zarr} · ` +
      `dask ${info.dask} · multiview-stitcher ${info.multiview_stitcher} · ` +
      `build ${build}`,
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

  claimTab();

  const { examples } = await command("examples", {});
  if (examples.length) {
    $("#example").textContent = `Load example: ${examples[0].label}`;
    $("#example").dataset.example = examples[0].name;
    $("#example").disabled = false;
  }

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
      setStatus(
        hasViews()
          ? "could not open that folder; the loaded views are unchanged"
          : "failed to open the dropped folder",
      );
    }
  });

  $("#browse").addEventListener("click", async () => {
    try {
      const handle = await window.showDirectoryPicker({ mode: "read" });
      await withPool(() => loadDirectory(handle));
    } catch (error) {
      if (error.name === "AbortError") return;
      log(error.message, "error");
      setStatus(
        hasViews()
          ? "could not open that folder; the loaded views are unchanged"
          : "failed to open the folder",
      );
    }
  });

  $("#transform-key").addEventListener("change", async (event) => {
    state.transformKey = event.target.value;
    log(`showing transform key '${state.transformKey}'`);
    await refreshViewer();
  });

  $("#example").addEventListener("click", async () => {
    try {
      await withPool(() => loadExample($("#example").dataset.example));
    } catch (error) {
      log(error.message, "error");
      setStatus("failed to load the example");
      setBusy(false);
    }
  });

  $("#clear").addEventListener("click", async () => {
    try {
      await clearSession();
    } catch (error) {
      log(error.message, "error");
    }
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
