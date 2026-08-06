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

// index.html loads this module as `app.js?v=<build>`; the same tag is passed
// to viewer.js so the two can never come from different builds.
const { NeuroglancerViewer } = await import(
  `./viewer.js${new URL(import.meta.url).search}`
);

const APP_BASE = new URL(".", window.location.href).pathname;
const API_BASE = `${APP_BASE}__mvs__`;

const state = {
  config: null,
  session: null, // most recent `describe()` result
  sessionSpec: null, // snapshot compute workers rebuild from
  position: null, // where the user is looking, reported by the viewer
  previewRoute: null,
  previewMetadata: null,
  previewTransformKey: null,
  previewVisibility: true,
  transformKey: null,
  layerSources: null, // name -> url of what the viewer currently shows
  currentViewLayerUrls: new Map(), // input source url -> viewer layer urls
  currentFusedLayerUrls: new Set(), // fused preview viewer layer urls
  viewVisibility: new Map(), // input source url -> visible
  channelVisibility: new Map(), // channel key -> visible
  selectedViewUrls: new Set(), // input source urls picked in the views list
  positionalColors: false,
  manualPlacement: false,
  placementChannels: new Map(), // channel key -> placement applies to it
  placementTimeRange: null, // [first, last] timepoint indices, inclusive
  placementTimeCount: null, // timepoints the range above was chosen for
  timeIndex: 0, // timepoint on screen, which is the one transforms are shown at
  timeVaryingTransforms: false, // whether following the timepoint is worth it
  editableTransformKeys: new Set(),
  mounts: [],
  // Local files every Python worker mounts for itself, as `{ mount, files }`.
  // Unlike an OME-Zarr - which Python reads over the service worker - a CZI is
  // opened as a path, so the handles have to be in each worker's own
  // filesystem. Kept here so that workers started later get the same mounts.
  fileMounts: [],
};

let displayVisibilityTimer = null;
let timeRefreshTimer = null;
let contrastSyncTimers = [];
let positionalColorRequest = 0;
let positionalColorTimers = [];

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
  const count = box.childElementCount;
  $("#log-count").textContent = String(count);
  $("#log-count").hidden = count === 0;

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

/**
 * Show how far a long job has got.
 *
 * The work runs inside one blocking call in the session worker, so the page
 * cannot observe it directly. It learns instead from each batch of tasks the
 * pool is handed, which is why work is dispatched in batches at all.
 */
function setProgress(progress) {
  const container = $("#progress");
  if (!progress || !progress.total) {
    container.hidden = true;
    $("#progress-bar").style.width = "0";
    $("#progress-percent").textContent = "";
    $("#status").hidden = false;
    return;
  }

  const { label, unit, completed, total } = progress;
  const fraction = Math.max(0, Math.min(1, completed / total));

  container.hidden = false;
  $("#status").hidden = true;
  $("#progress-bar").style.width = `${(fraction * 100).toFixed(1)}%`;
  $("#progress-percent").textContent = `${Math.round(fraction * 100)}%`;
  $("#progress-label").textContent =
    `${label} ${completed}/${total} ${unit}${total === 1 ? "" : "s"}`;
}

function clearProgress() {
  setProgress(null);
}

/**
 * How far the Python runtimes have got.
 *
 * Booting is the longest wait in the app - tens of seconds, most of it
 * downloading - and it happens before there is anything else to look at, so
 * the workers report which step they are on and the bar shows the total.
 */
const bootPhases = new Map();

function noteBootPhase(worker, phase, phases) {
  bootPhases.set(worker, { phase, phases });

  let completed = 0;
  let total = 0;
  for (const entry of bootPhases.values()) {
    completed += entry.phase;
    total += entry.phases;
  }

  if (completed >= total) {
    // Every runtime that has started is up. A pool that grows later starts
    // its own count rather than reviving this one.
    bootPhases.clear();
    clearProgress();
    return;
  }

  setProgress({
    label: "starting Python",
    unit: "step",
    completed,
    total,
  });
}

function setBusy(busy) {
  for (const button of document.querySelectorAll("button[data-action]")) {
    // Loading actions stay available while there is no data; the processing
    // ones need views first.
    const needsData = button.dataset.needsData !== "false";
    button.disabled = busy || (needsData && !hasViews());
  }
  for (const button of document.querySelectorAll("button[data-load]")) {
    button.disabled = busy || (button.id === "clear" && !hasViews());
  }
  $("#worker-count").disabled = busy;
}

/** Promise-based request/response over a Worker's postMessage. */
class WorkerChannel {
  constructor(url, name) {
    // Module workers throughout: Pyodide will not start in a classic one.
    this.worker = new Worker(url, { type: "module" });
    this.name = name;
    this.pending = new Map();
    this.nextId = 1;
    this.busy = false;

    this.worker.onmessage = (event) => {
      const { id, type } = event.data;
      if (type === "log") {
        log(`${name}: ${event.data.message}`);
        if (event.data.phase) {
          noteBootPhase(name, event.data.phase, event.data.phases);
        }
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
    this.pending = null;
  }

  get size() {
    return this.workers.length;
  }

  /**
   * Grow or shrink the pool to `count` booted workers.
   *
   * Resizes are serialised through `this.pending` so that starting workers in
   * the background cannot race a later change of mind; callers that need a
   * ready pool await `ready()` instead of resizing again.
   */
  resize(count, config) {
    this.pending = Promise.resolve(this.pending)
      .catch(() => {})
      .then(() => this._resize(count, config));
    return this.pending;
  }

  /** Resolves once any in-flight resize has finished. */
  async ready() {
    await Promise.resolve(this.pending).catch(() => {});
  }

  async _resize(count, config) {
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
          // A worker started after files were mounted has to be given them
          // too, before it is handed a task that opens one by path.
          .then(() => Promise.all(
            state.fileMounts.map((entry) =>
              channel.send({
                type: "mount-files",
                mount: entry.mount,
                files: entry.files,
              }),
            ),
          ))
          .then(() => {
            channel.busy = false;
            this.pump();
          }),
      );
    }

    if (booting.length) {
      log(`starting ${booting.length} compute worker(s)`);
      await Promise.all(booting);
      log(`${this.workers.length} compute worker(s) ready`);
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
      if (payload.progress) setProgress(payload.progress);

      const results = await pool.dispatch(payload.tasks || [], state.sessionSpec);

      // Python can only report progress when it dispatches, so the last
      // batch's completion never arrives on its own - and a job that fits in
      // a single batch would sit at 0% and then disappear. This batch is
      // finished now, so count it here.
      if (payload.progress && payload.progress.batch) {
        setProgress({
          ...payload.progress,
          completed: payload.progress.completed + payload.progress.batch,
        });
      }

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

const viewer = new NeuroglancerViewer();
let placementSyncTimer = null;
let placementSync = Promise.resolve();
const placementSignatures = new Map();

function editedTransforms(ngState) {
  const byIndex = new Map();
  for (const layer of ngState.layers || []) {
    const match = String(layer.name || "").match(/^(\d+):/);
    if (!match || byIndex.has(Number(match[1]))) continue;
    const sourceSpecs = Array.isArray(layer.source) ? layer.source : [layer.source];
    const source = sourceSpecs.find(
      (candidate) => candidate && typeof candidate === "object" && candidate.transform,
    );
    if (source) {
      byIndex.set(Number(match[1]), {
        index: Number(match[1]),
        transform: source.transform,
      });
    }
  }
  return Array.from(byIndex.values()).sort((a, b) => a.index - b.index);
}

function schedulePlacementSync(ngState, { fromDrag = false } = {}) {
  const transformKey = state.transformKey;
  // A manual placement is saved into whatever coordinate system is displayed -
  // that is what the user was looking at when they moved the tile. An edit
  // made through Neuroglancer's own source tab has no such moment to point
  // at, so it is only followed for the keys created here, as before.
  if (!fromDrag && !state.editableTransformKeys.has(transformKey)) return;
  // A tile in mid-drag is not a placement yet. The layer follows the pointer
  // through the viewer state, so without this every frame of a drag would be a
  // coordinate system to save - and each save retires the fused preview. The
  // drag ends by calling here again, with `dragging` already false.
  if (viewer.dragging) return;

  const updates = editedTransforms(ngState);
  if (!updates.length) return;
  // A drag carries the scope it was made under, so a later change to the
  // checkboxes cannot rewrite what was already saved.
  const channels = fromDrag ? placementChannels() : null;
  const timeRange = fromDrag ? placementTimeRange() : null;
  const signature = JSON.stringify([updates, channels, timeRange]);
  if (placementSignatures.get(transformKey) === signature) return;

  clearTimeout(placementSyncTimer);
  placementSyncTimer = setTimeout(() => {
    placementSync = placementSync
      .catch(() => {})
      .then(async () => {
        if (placementSignatures.get(transformKey) === signature) return;
        if (viewer.dragging) return;
        placementSignatures.set(transformKey, signature);
        try {
          const result = await command("update_transforms", {
            transform_key: transformKey,
            updates,
            channels: channels
              ? channels.map((index) => displayChannels()[index]?.name)
              : null,
            time_range: timeRange,
          });
          if (state.session) state.session.generation = result.generation;
          // From here on the viewer has to follow the timepoint: what it shows
          // is one sample of a transform that now differs between them.
          if (timeRange) state.timeVaryingTransforms = true;
          const hadPreview = Boolean(state.previewRoute);
          state.previewRoute = null;
          state.previewMetadata = null;
          state.previewTransformKey = null;
          await refreshSessionSpec();
          // A drag can only show one timepoint's worth of a placement that
          // was stored for a range of them, so the viewer is rebuilt from
          // what the session actually holds rather than left showing the
          // drag. The channel restriction needs no such correction: those
          // layers moved and the others did not.
          if (hadPreview || timeRange) {
            await refreshViewer();
            renderViews(state.session);
          }
          setStatus(`saved placement in ${transformKey}`);
        } catch (error) {
          placementSignatures.delete(transformKey);
          log(`could not save tile placement: ${error.message}`, "error");
        }
      });
  }, 350);
}

function mountViewer() {
  if (viewer.mounted) return;

  viewer.mount($("#viewer"));

  // The viewer is the source of truth for where the user is looking; the app
  // only needs to know when that changes.
  viewer.onPositionChanged((position) => {
    state.position = position;
    noteTimeIndex();
  });
  viewer.onStateChanged((ngState) => {
    schedulePlacementSync(ngState);
    scheduleDisplayVisibility();
  });
}

/**
 * Follow the timepoint being viewed, refreshing when it moves.
 *
 * Only worth doing once a transform actually varies over time: a source
 * transform is one matrix, so the state has to be rebuilt to show a different
 * timepoint's, and doing that on every scrub of a dataset whose placement is
 * the same throughout would be pure cost.
 */
function noteTimeIndex() {
  if (timeCoords().length < 2) return;

  const dimensions = viewer.getPositionDimensions();
  const axis = dimensions.indexOf("t");
  if (axis < 0) return;

  const value = Math.round(state.position?.[axis] ?? 0);
  const index = Math.min(Math.max(value, 0), timeCoords().length - 1);
  if (index === state.timeIndex) return;

  state.timeIndex = index;
  renderTimeSlider();
  if (!state.timeVaryingTransforms) return;

  clearTimeout(timeRefreshTimer);
  timeRefreshTimer = setTimeout(() => {
    refreshViewer().catch((error) =>
      log(`could not follow the timepoint: ${error.message}`, "warn"),
    );
  }, 200);
}

/** The layers a state describes, as `name -> source url`. */
function layerSources(ngState) {
  const sources = {};
  for (const layer of ngState.layers || []) {
    const source = layer.source;
    sources[layer.name] = typeof source === "string" ? source : source.url;
  }
  return sources;
}

function showViewerState(ngState) {
  mountViewer();

  const sources = layerSources(ngState);

  // A transform_key switch changes only where each layer sits. Handing the
  // whole state over would rebuild every layer, taking the user's shader
  // settings, the selected layer and the chosen layout with it - so when the
  // layers themselves are unchanged, only their transforms and visibility
  // are applied.
  const known = state.layerSources ? Object.values(state.layerSources) : [];
  const wanted = Object.values(sources);

  // Keyed by source URL, not by layer name: Neuroglancer renames layers it
  // opens - an OME-Zarr with omero metadata becomes "<name> channel 0" - and
  // a lookup by name would miss, fall back to applying the whole state and
  // take the layout and the shader settings with it.
  const transforms = {};
  const visibility = {};
  for (const layer of ngState.layers || []) {
    const source = layer.source;
    const url = typeof source === "string" ? source : source.url;
    transforms[url] =
      typeof source === "string" ? null : source.transform || null;
    visibility[url] = layer.visible !== false;
  }

  // Nothing in common with what is on screen means a different dataset, not
  // an edit of this one. Its coordinate space, camera and layout say nothing
  // about the new data, so the viewer starts clean and Neuroglancer places
  // the camera on what actually loaded.
  if (state.layerSources && !wanted.some((url) => known.includes(url))) {
    viewer.reset();
    state.layerSources = sources;
    viewer.setState(ngState);
    return;
  }

  if (state.layerSources) {
    // Same dataset. Whether layers only moved, or one was added or removed -
    // the fused preview appearing is the usual case - the difference is
    // applied on its own. Restoring a `layers` array instead would clear the
    // list and rebuild every layer, replacing the chosen layout, the selected
    // layer and each layer's shader and contrast range.
    const added = (ngState.layers || []).filter((layer) => {
      const source = layer.source;
      const url = typeof source === "string" ? source : source.url;
      return !known.includes(url);
    });
    const removed = known.filter((url) => !wanted.includes(url));

    try {
      if (removed.length) viewer.removeLayers(removed);
      if (added.length) viewer.addLayers(added);

      // Only the layers that were already there: a layer added a moment ago
      // has no loaded source to re-aim yet, and carries its transform in the
      // specification it was built from.
      const survivors = Object.fromEntries(
        Object.entries(transforms).filter(([url]) => known.includes(url)),
      );
      viewer.setLayerTransforms(survivors);
      viewer.setLayerVisibility(
        Object.fromEntries(
          Object.entries(visibility).filter(([url]) => known.includes(url)),
        ),
      );

      state.layerSources = sources;
      return;
    } catch (error) {
      // The viewer's layers are not the ones we last applied - the user can
      // rename or remove them in Neuroglancer's own UI. Fall back to applying
      // the whole state rather than leaving a half-updated view.
      log(`re-applying the full viewer state: ${error.message}`, "warn");
    }
  }

  state.layerSources = sources;
  // Applied to the running viewer: the camera, the WebGL context and anything
  // already fetched all survive, so switching transform_key is immediate.
  viewer.setState(ngState);
}

function applyViewVisibility(ngState) {
  state.currentViewLayerUrls = new Map();
  state.currentFusedLayerUrls = new Set();
  for (const layer of ngState.layers || []) {
    const source = layer.source;
    const url = typeof source === "string" ? source : source.url;
    if (String(layer.name || "").startsWith("fused")) {
      state.currentFusedLayerUrls.add(url);
      continue;
    }
    const match = String(layer.name || "").match(/^(\d+):/);
    if (!match) continue;
    const index = Number(match[1]);
    const view = state.session?.views?.[index];
    if (!view) continue;
    if (!state.currentViewLayerUrls.has(view.url)) {
      state.currentViewLayerUrls.set(view.url, new Set());
    }
    state.currentViewLayerUrls.get(view.url).add(url);
    layer.visible = state.viewVisibility.get(view.url) !== false;
  }
}

function fusedLayerVisibility() {
  const visible =
    state.previewVisibility &&
    (!state.previewTransformKey || state.previewTransformKey === state.transformKey);
  return Object.fromEntries(
    [...state.currentFusedLayerUrls].map((url) => [url, visible]),
  );
}

function inputLayerVisibility() {
  const visibility = {};
  for (const [viewUrl, layerUrls] of state.currentViewLayerUrls) {
    for (const layerUrl of layerUrls) {
      visibility[layerUrl] = state.viewVisibility.get(viewUrl) !== false;
    }
  }
  return visibility;
}

function displayChannels() {
  const channels = state.session?.views?.[0]?.c_coords || [];
  return channels.length
    ? channels.map((name, index) => ({ name: String(name), index, key: String(index) }))
    : [{ name: "Image", index: null, key: "default" }];
}

function channelVisibility() {
  return Object.fromEntries(
    displayChannels().map(({ key, index }) => [
      index === null ? "default" : index,
      state.channelVisibility.get(key) !== false,
    ]),
  );
}

function applyDisplayVisibility() {
  if (!viewer.mounted) return 0;
  return viewer.setDisplayVisibility(
    {
      ...inputLayerVisibility(),
      ...fusedLayerVisibility(),
    },
    channelVisibility(),
  );
}

async function applyPositionalColors() {
  const request = ++positionalColorRequest;
  for (const timer of positionalColorTimers) clearTimeout(timer);
  positionalColorTimers = [];
  if (!viewer.mounted || !state.positionalColors || !hasViews()) {
    if (viewer.mounted) viewer.setPositionalColors(null);
    return;
  }

  const result = await command("positional_colors", {
    transform_key: state.transformKey,
    n_colors: 2,
  });
  if (request !== positionalColorRequest || !state.positionalColors) return;

  const colors = {};
  state.session.views.forEach((view, index) => {
    for (const url of state.currentViewLayerUrls.get(view.url) || []) {
      colors[url] = result.colors[index];
    }
  });
  viewer.setPositionalColors(colors);
  positionalColorTimers = [700, 1600, 3000].map((delay) =>
    setTimeout(() => {
      if (request === positionalColorRequest && state.positionalColors) {
        viewer.setPositionalColors(colors);
      }
    }, delay),
  );
}

function scheduleDisplayVisibility(delay = 0) {
  clearTimeout(displayVisibilityTimer);
  displayVisibilityTimer = setTimeout(() => {
    try {
      applyDisplayVisibility();
    } catch (error) {
      log(`could not update layer visibility: ${error.message}`, "warn");
    }
  }, delay);
}

function formatContrast(value) {
  if (!Number.isFinite(value)) return "–";
  return Number(value.toPrecision(6)).toString();
}

function updateContrastFill(row) {
  const low = row.querySelector(".contrast-min-slider");
  const high = row.querySelector(".contrast-max-slider");
  const span = Number(high.max) - Number(high.min);
  if (!(span > 0)) return;
  const lowPercent = ((Number(low.value) - Number(low.min)) / span) * 100;
  const highPercent = ((Number(high.value) - Number(high.min)) / span) * 100;
  const range = row.querySelector(".dual-range");
  range.style.setProperty("--low", `${lowPercent}%`);
  range.style.setProperty("--high", `${highPercent}%`);
}

function setContrastUi(row, { min, max, lower = min, upper = max }) {
  lower = Math.min(lower, min);
  upper = Math.max(upper, max);
  if (!(upper > lower)) {
    const padding = Math.max(Math.abs(lower) * 0.01, 1);
    lower -= padding;
    upper += padding;
  }
  const step = Math.max((upper - lower) / 1000, Number.EPSILON);
  const minSlider = row.querySelector(".contrast-min-slider");
  const maxSlider = row.querySelector(".contrast-max-slider");
  for (const slider of [minSlider, maxSlider]) {
    slider.min = String(lower);
    slider.max = String(upper);
    slider.step = String(step);
  }
  minSlider.value = String(min);
  maxSlider.value = String(max);
  row.querySelector(".contrast-min-value").value = formatContrast(min);
  row.querySelector(".contrast-max-value").value = formatContrast(max);
  updateContrastFill(row);
}

function syncContrastUi() {
  if (!viewer.mounted || !hasViews()) return false;
  let synced = 0;
  for (const row of document.querySelectorAll(".channel-control")) {
    const channelIndex = row.dataset.channelIndex === ""
      ? null
      : Number(row.dataset.channelIndex);
    const limits = viewer.getContrastLimits(
      inputLayerVisibility(),
      channelIndex,
    );
    if (!limits) continue;
    setContrastUi(row, limits);
    if (state.currentFusedLayerUrls.size) {
      viewer.setContrastLimits(
        Object.fromEntries(
          [...state.currentFusedLayerUrls].map((url) => [url, true]),
        ),
        channelIndex,
        [limits.min, limits.max],
      );
    }
    synced += 1;
  }
  return synced > 0;
}

function scheduleContrastSync() {
  for (const timer of contrastSyncTimers) clearTimeout(timer);
  contrastSyncTimers = [0, 1400, 2800].map((delay) =>
    setTimeout(() => syncContrastUi(), delay),
  );
}

function applyContrastLimits(row, channelIndex, min, max) {
  if (!Number.isFinite(min) || !Number.isFinite(max) || !(min < max)) {
    throw new Error("Enter two numbers with the minimum below the maximum.");
  }
  const minSlider = row.querySelector(".contrast-min-slider");
  const maxSlider = row.querySelector(".contrast-max-slider");
  setContrastUi(row, {
    min,
    max,
    lower: Math.min(min, Number(minSlider.min)),
    upper: Math.max(max, Number(maxSlider.max)),
  });
  const changed = viewer.setContrastLimits(
    {
      ...inputLayerVisibility(),
      ...Object.fromEntries(
        [...state.currentFusedLayerUrls].map((url) => [url, true]),
      ),
    },
    channelIndex,
    [min, max],
  );
  if (!changed) {
    setStatus("contrast controls are still loading");
    scheduleContrastSync();
  }
}

async function refreshViewer() {
  if (!hasViews()) {
    // No views: back to a blank viewer, not an empty layer list on top of the
    // previous dataset's coordinate space.
    if (viewer.mounted) viewer.reset();
    state.layerSources = null;
    state.currentViewLayerUrls.clear();
    $("#viewer-empty").hidden = false;
    syncManualPlacement();
    return;
  }

  const geometry = {
    transform_key: state.transformKey,
    base_url: window.location.origin,
    // The service worker only claims URLs inside its own scope, so Python
    // must build viewer URLs below this prefix rather than at the site root.
    api_base: API_BASE,
    // A source transform is one matrix, so it can only be one timepoint's.
    // Showing the one being viewed is what makes a placement stored for a
    // range of them legible.
    time_index: state.timeIndex,
  };
  const ngState = await command("neuroglancer_state", {
    ...geometry,
    preview_route: state.previewRoute,
  });

  applyViewVisibility(ngState);
  $("#viewer-empty").hidden = true;
  showViewerState(ngState);
  await applyChannelTransforms(geometry);
  applyDisplayVisibility();
  await applyPositionalColors();
  scheduleDisplayVisibility(500);
  scheduleContrastSync();
  // Layers may have been added, removed or rebuilt, and placement addresses
  // them by URL - so it is re-applied against whatever is on screen now.
  syncManualPlacement();
}

/**
 * Aim the per-channel layers of any view whose transform varies by channel.
 *
 * A Neuroglancer layer carries one source transform and the viewer opens one
 * layer per channel, so a channel-dependent placement cannot travel inside the
 * state - it arrives beside it. Nothing is sent for the ordinary case, where
 * every channel shares a transform.
 */
async function applyChannelTransforms(geometry) {
  let transforms;
  try {
    transforms = await command("channel_transforms", geometry);
  } catch (error) {
    log(`could not read per-channel transforms: ${error.message}`, "warn");
    return;
  }
  if (!transforms || !Object.keys(transforms).length) return;

  try {
    viewer.setChannelTransforms(transforms);
  } catch (error) {
    log(`could not apply per-channel transforms: ${error.message}`, "warn");
  }
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

function renderDimensionFields(described) {
  const view = described.views[0];
  const dims = view?.spatial_dims || [];
  const defaults = { z: 3, y: 10, x: 10 };

  for (const [containerSelector, prefix, values] of [
    ["#registration-binning", "registration-binning", {}],
    ["#blending-widths", "blending-width", defaults],
    ["#output-spacing", "output-spacing", {}],
    ["#output-chunksizes", "output-chunksize", {}],
  ]) {
    const container = $(containerSelector);
    container.replaceChildren();
    for (const dim of dims) {
      const wrapper = document.createElement("div");
      const label = document.createElement("label");
      const input = document.createElement("input");
      const id = `${prefix}-${dim}`;
      label.htmlFor = id;
      label.textContent = dim;
      input.id = id;
      input.type = "number";
      const integer = ["registration-binning", "output-chunksize"].includes(prefix);
      input.min = integer ? "1" : "0";
      input.step = integer ? "1" : "any";
      input.dataset.dimension = dim;
      input.value = values[dim] ?? "";
      input.placeholder = "auto";
      wrapper.append(label, input);
      container.appendChild(wrapper);
    }
  }
}

function renderChannelControls(described) {
  const channels = described.views[0]?.c_coords || [];
  const controls = $("#channel-controls");
  controls.replaceChildren();
  $("#empty-channels").hidden = Boolean(described.n_views);
  $("#positional-colors").disabled = !described.n_views;
  $("#positional-colors").checked = state.positionalColors;

  if (described.n_views) {
    const activeKeys = new Set(displayChannels().map(({ key }) => key));
    for (const key of state.channelVisibility.keys()) {
      if (!activeKeys.has(key)) state.channelVisibility.delete(key);
    }

    for (const { name, index, key } of displayChannels()) {
      if (!state.channelVisibility.has(key)) state.channelVisibility.set(key, true);

      const row = document.createElement("div");
      row.className = "channel-control";
      row.dataset.channelIndex = index === null ? "" : String(index);

      const heading = document.createElement("label");
      heading.className = "channel-heading";
      const visibility = document.createElement("input");
      visibility.type = "checkbox";
      visibility.className = "visibility-toggle channel-visibility";
      visibility.checked = state.channelVisibility.get(key) !== false;
      visibility.setAttribute("aria-label", `Show channel ${name}`);
      const channelName = document.createElement("span");
      channelName.textContent = name;
      heading.append(visibility, channelName);

      const line = document.createElement("div");
      line.className = "channel-contrast-line";
      const minValue = document.createElement("input");
      minValue.type = "text";
      minValue.inputMode = "decimal";
      minValue.className = "contrast-min-value";
      minValue.value = "0";
      minValue.placeholder = "min";
      minValue.setAttribute("aria-label", `Minimum contrast for ${name}`);

      const range = document.createElement("div");
      range.className = "dual-range";
      range.setAttribute("aria-label", `Contrast limits for ${name}`);
      const track = document.createElement("div");
      track.className = "dual-range-track";
      const minSlider = document.createElement("input");
      minSlider.type = "range";
      minSlider.className = "contrast-min-slider";
      minSlider.min = "0";
      minSlider.max = "1";
      minSlider.step = "0.001";
      minSlider.value = "0";
      minSlider.setAttribute("aria-label", `Minimum contrast slider for ${name}`);
      const maxSlider = document.createElement("input");
      maxSlider.type = "range";
      maxSlider.className = "contrast-max-slider";
      maxSlider.min = "0";
      maxSlider.max = "1";
      maxSlider.step = "0.001";
      maxSlider.value = "1";
      maxSlider.setAttribute("aria-label", `Maximum contrast slider for ${name}`);
      range.append(track, minSlider, maxSlider);

      const maxValue = document.createElement("input");
      maxValue.type = "text";
      maxValue.inputMode = "decimal";
      maxValue.className = "contrast-max-value";
      maxValue.value = "1";
      maxValue.placeholder = "max";
      maxValue.setAttribute("aria-label", `Maximum contrast for ${name}`);
      line.append(minValue, range, maxValue);
      row.append(heading, line);
      controls.appendChild(row);

      visibility.addEventListener("change", () => {
        state.channelVisibility.set(key, visibility.checked);
        log(`${visibility.checked ? "showing" : "hiding"} channel '${name}'`);
        applyDisplayVisibility();
        scheduleDisplayVisibility(300);
      });

      for (const slider of [minSlider, maxSlider]) {
        slider.addEventListener("input", (event) => {
          let min = Number(minSlider.value);
          let max = Number(maxSlider.value);
          const step = Number(event.target.step) || Number.EPSILON;
          if (event.target === minSlider && min >= max) min = max - step;
          if (event.target === maxSlider && max <= min) max = min + step;
          try {
            applyContrastLimits(row, index, min, max);
          } catch (error) {
            setStatus(error.message);
          }
        });
      }

      const commitValues = (event) => {
        try {
          applyContrastLimits(row, index, Number(minValue.value), Number(maxValue.value));
        } catch (error) {
          setStatus(error.message);
          event.target.select();
        }
      };
      minValue.addEventListener("change", commitValues);
      maxValue.addEventListener("change", commitValues);
    }
  }

  const registration = $("#registration-channel");
  registration.replaceChildren();
  if (channels.length) {
    channels.forEach((channel, index) =>
      registration.add(new Option(channel, String(index))),
    );
  } else {
    registration.add(new Option("No channel axis", ""));
  }
  registration.disabled = !described.n_views || channels.length < 2;
}

/**
 * Whether every input view is shown.
 *
 * The fused preview is deliberately not counted: it is derived data with its
 * own toggle, and switching the inputs off to look at the fusion is the whole
 * point of the "hide all" button.
 */
function allViewsVisible(views) {
  return views.every((view) => state.viewVisibility.get(view.url) !== false);
}

function renderViewToggle(views) {
  const toggle = $("#toggle-views");
  const showAll = !allViewsVisible(views);
  toggle.hidden = !views.length;
  toggle.textContent = showAll ? "Show all" : "Hide all";
  toggle.title = showAll
    ? "Show every input view"
    : "Hide every input view, keeping the fused preview";

  const selection = $("#toggle-selection");
  const allSelected =
    views.length > 0 &&
    views.every((view) => state.selectedViewUrls.has(view.url));
  selection.hidden = !views.length;
  selection.textContent = allSelected ? "Select none" : "Select all";
  selection.title = allSelected
    ? "Clear the selection"
    : "Select every view, so a drag moves them together";
}

// ---------------------------------------------------------------------------
// Manual placement
// ---------------------------------------------------------------------------

/**
 * Pick the views a ctrl+drag acts on.
 *
 * One selected view breaks the tie where tiles overlap; several make a drag
 * move all of them together. Nothing is recomputed and no data moves: the
 * selection only ever decides which layers a drag resolves to, so the viewer
 * is simply told about it.
 *
 * A plain click selects one view, replacing whatever was selected, and
 * clicking the only selected view clears it. Ctrl+click adds and removes,
 * which is how a set is built up.
 */
function selectView(url, { extend = false } = {}) {
  const selected = state.selectedViewUrls;

  if (extend) {
    if (selected.has(url)) selected.delete(url);
    else selected.add(url);
  } else if (selected.size === 1 && selected.has(url)) {
    selected.clear();
  } else {
    selected.clear();
    selected.add(url);
  }

  renderViews(state.session);
  syncManualPlacement();
}

/** Select every view, or none if they are already all selected. */
function toggleAllSelected() {
  const views = state.session?.views || [];
  if (!views.length) return;

  const all = views.every((view) => state.selectedViewUrls.has(view.url));
  state.selectedViewUrls.clear();
  if (!all) {
    for (const view of views) state.selectedViewUrls.add(view.url);
  }

  renderViews(state.session);
  syncManualPlacement();
}

/** The viewer layers a view is drawn as - one per channel. */
function layerUrlsOf(viewUrl) {
  return Array.from(state.currentViewLayerUrls.get(viewUrl) || []);
}

/** The timepoints the loaded data has, as their coordinate values. */
function timeCoords() {
  return state.session?.views?.[0]?.t_coords || [];
}

/**
 * The channels a placement applies to, as indices, or null for all of them.
 *
 * Null is the case worth distinguishing: it lets the session store one affine
 * for the whole image rather than one per channel, which is what keeps the
 * parameters free of a channel axis they would not vary over.
 */
function placementChannels() {
  const channels = displayChannels().filter(({ index }) => index !== null);
  if (!channels.length) return null;
  const chosen = channels.filter(
    ({ key }) => state.placementChannels.get(key) !== false,
  );
  return chosen.length === channels.length
    ? null
    : chosen.map(({ index }) => index);
}

/** The timepoint range a placement applies to, or null for all of them. */
function placementTimeRange() {
  const count = timeCoords().length;
  if (count < 2) return null;
  const [first, last] = state.placementTimeRange ?? [0, count - 1];
  return first === 0 && last === count - 1 ? null : [first, last];
}

/**
 * Rebuild the channel checkboxes and the timepoint slider for the loaded data.
 *
 * Both start unrestricted - every channel, every timepoint - so that a user
 * who never opens this section places tiles the way they always did.
 */
function renderPlacementScope(described) {
  const channels = displayChannels().filter(({ index }) => index !== null);
  const container = $("#placement-channels");
  container.replaceChildren();
  $("#placement-channels-empty").hidden = channels.length > 0;
  container.hidden = channels.length === 0;

  // A scope belongs to the dataset it was chosen for; a different one starts
  // unrestricted rather than inheriting a selection nothing on screen shows.
  const keys = new Set(channels.map(({ key }) => key));
  for (const key of state.placementChannels.keys()) {
    if (!keys.has(key)) state.placementChannels.delete(key);
  }

  for (const { name, key } of channels) {
    if (!state.placementChannels.has(key)) {
      state.placementChannels.set(key, true);
    }
    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = state.placementChannels.get(key) !== false;
    box.addEventListener("change", () => {
      // Never leave every channel unticked: a placement that applies to
      // nothing would be a drag the viewer refuses for no visible reason.
      const others = channels.filter((channel) => channel.key !== key);
      const anyOther = others.some(
        (channel) => state.placementChannels.get(channel.key) !== false,
      );
      if (!box.checked && !anyOther) {
        box.checked = true;
        setStatus("at least one channel has to be placed");
        return;
      }
      state.placementChannels.set(key, box.checked);
      syncManualPlacement();
    });
    const text = document.createElement("span");
    text.textContent = name;
    label.append(box, text);
    container.appendChild(label);
  }

  const times = timeCoords();
  const timeBlock = $("#placement-time").closest(".scope-block");
  const hasTime = times.length > 1;
  timeBlock.classList.toggle("disabled", !hasTime);
  $("#placement-time").hidden = !hasTime;
  $("#placement-time-empty").hidden = hasTime;

  const first = $("#placement-time-first");
  const last = $("#placement-time-last");
  for (const slider of [first, last]) {
    slider.max = String(Math.max(times.length - 1, 0));
    slider.disabled = !hasTime;
  }
  if (state.placementTimeCount !== times.length) {
    state.placementTimeCount = times.length;
    state.placementTimeRange = null;
  }
  if (!state.placementTimeRange) {
    state.placementTimeRange = [0, Math.max(times.length - 1, 0)];
  }
  first.value = String(state.placementTimeRange[0]);
  last.value = String(state.placementTimeRange[1]);
  updatePlacementTimeUi();

  $("#placement-scope").hidden = !described.n_views;
}

/**
 * Show the time slider when the data has a time axis, and put it where the
 * viewer is looking.
 *
 * Neuroglancer has no notion of a current frame - time is one axis of the
 * position - so this is a control over the position, not a mode.
 */
function renderTimeSlider() {
  const times = timeCoords();
  const section = $("#time-section");
  const slider = $("#time-slider");

  section.hidden = times.length < 2;
  if (section.hidden) return;

  slider.max = String(times.length - 1);
  state.timeIndex = Math.min(state.timeIndex, times.length - 1);
  slider.value = String(state.timeIndex);
  $("#time-value").textContent = `${times[state.timeIndex]} of ${
    times[times.length - 1]
  }`;
}

/** Reflect the timepoint range in the slider fill and the heading. */
function updatePlacementTimeUi() {
  const times = timeCoords();
  const row = $("#placement-time");
  const [first, last] = state.placementTimeRange ?? [0, 0];
  const span = Math.max(times.length - 1, 1);
  row.style.setProperty("--low", `${(first / span) * 100}%`);
  row.style.setProperty("--high", `${(last / span) * 100}%`);
  $("#placement-time-label").textContent = times.length
    ? first === last
      ? `${times[first]}`
      : `${times[first]} – ${times[last]}`
    : "";
}

/**
 * Hand the viewer the current placement settings, or switch placement off.
 *
 * Called whenever anything it depends on moves: the checkbox, the selected
 * view, the transform key, and every viewer refresh - a refresh can rebuild
 * the layers, and the URLs are how the viewer addresses them.
 */
function syncManualPlacement() {
  const possible = hasViews();
  const checkbox = $("#manual-placement");
  checkbox.disabled = !possible;
  if (!possible) checkbox.checked = false;

  const active = possible && checkbox.checked;
  state.manualPlacement = active;

  // The tiles a drag acts on are chosen in the views list rather than here, so
  // the section says what that selection currently amounts to.
  const selected = state.selectedViewUrls.size;
  $("#placement-tiles-label").textContent =
    selected > 1
      ? `${selected} selected`
      : selected === 1
        ? "1 selected"
        : "whichever is under the pointer";
  // Placement writes into whatever coordinate system is on screen, including
  // the one an OME-Zarr came with, so the panel says which that is rather than
  // leaving the user to infer it from the other side of the window.
  const target = $("#manual-placement-target");
  target.hidden = !active;
  target.textContent = `Drags are saved into “${state.transformKey}”.`;

  if (!viewer.mounted) return;
  if (!active) {
    viewer.setManualPlacement(null);
    return;
  }

  viewer.setManualPlacement({
    // Input views only: a fused image is derived from where the tiles are, so
    // dragging it would describe nothing the session can save.
    movableUrls: [...state.currentViewLayerUrls.keys()].flatMap(layerUrlsOf),
    // Several selected views make a drag move them all; one breaks a tie
    // where tiles overlap.
    selectedUrls: [...state.selectedViewUrls].flatMap(layerUrlsOf),
    // Only the chosen channels follow the pointer. A timepoint range cannot
    // be shown this way - one source transform covers the whole time axis -
    // so the drag shows the timepoint on screen and the session stores the
    // range, which is why the viewer is refreshed afterwards.
    channels: placementChannels(),
    onDragStart: (urls, mode) =>
      setStatus(
        `${mode === "rotate" ? "turning" : "moving"} ` +
          `${urls.length === 1 ? "a tile" : `${urls.length} tiles`} in ` +
          `${state.transformKey}`,
        true,
      ),
    onDragEnd: () =>
      schedulePlacementSync(viewer.getState(), { fromDrag: true }),
    onRefused: (reason) =>
      setStatus(
        {
          ambiguous:
            "several tiles here - select one in the views list to move it",
          "outside-selection":
            "start the drag on one of the selected tiles to move them",
          "no-channels": "no channel here is set to be placed",
        }[reason] ?? "no tile under the pointer",
      ),
  });
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

    // Selecting views is what tells a ctrl+drag which tiles to move: one
    // breaks a tie where they overlap, several move together. The whole row
    // is the target, minus its own controls.
    const selected = state.selectedViewUrls.has(view.url);
    item.className = "selectable";
    item.tabIndex = 0;
    item.setAttribute("role", "option");
    item.setAttribute("aria-selected", String(selected));
    if (selected) item.classList.add("selected");
    item.title = `Select ${view.name} for manual placement (ctrl+click to add)`;

    const select = (event) => {
      if (event.target.closest("input, button")) return;
      selectView(view.url, { extend: event.ctrlKey || event.metaKey });
    };
    item.addEventListener("click", select);
    item.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      select(event);
    });

    if (!state.viewVisibility.has(view.url)) {
      state.viewVisibility.set(view.url, true);
    }

    const visibility = document.createElement("input");
    visibility.type = "checkbox";
    visibility.className = "visibility-toggle";
    visibility.checked = state.viewVisibility.get(view.url) !== false;
    visibility.title = `Show or hide ${view.name}`;
    visibility.setAttribute("aria-label", `Show ${view.name}`);
    visibility.addEventListener("change", () => {
      state.viewVisibility.set(view.url, visibility.checked);
      applyDisplayVisibility();
    });

    const text = document.createElement("div");
    const name = document.createElement("strong");
    // The viewer names its layers the same way, so the two lists line up.
    name.textContent = `${index}: ${view.name}`;
    const detail = document.createElement("span");
    detail.className = "view-detail";
    detail.textContent = `${shape} · ${view.levels.length} level${view.levels.length === 1 ? "" : "s"}`;
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

    item.append(visibility, text, remove);
    list.appendChild(item);
  });

  if (state.previewRoute && state.previewMetadata) {
    const level = state.previewMetadata.levels[0];
    const shape = Object.entries(level.shape)
      .map(([dim, size]) => `${dim}:${size}`)
      .join(" ");
    const item = document.createElement("li");
    item.className = "derived-view";

    const visibility = document.createElement("input");
    visibility.type = "checkbox";
    visibility.className = "visibility-toggle";
    visibility.checked = state.previewVisibility;
    visibility.title = "Show or hide fused preview";
    visibility.setAttribute("aria-label", "Show fused preview");
    visibility.addEventListener("change", () => {
      state.previewVisibility = visibility.checked;
      applyDisplayVisibility();
    });

    const text = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = "Fused preview";
    const detail = document.createElement("span");
    detail.className = "view-detail";
    detail.textContent = `${shape} · ${state.previewMetadata.levels.length} level${state.previewMetadata.levels.length === 1 ? "" : "s"}`;
    text.append(name, detail);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "remove";
    remove.title = "Remove fused preview";
    remove.textContent = "×";
    remove.addEventListener("click", async () => {
      state.previewRoute = null;
      state.previewMetadata = null;
      state.previewTransformKey = null;
      state.currentFusedLayerUrls.clear();
      await refreshViewer();
      renderViews(state.session);
      setStatus("fused preview removed");
    });

    item.append(visibility, text, remove);
    list.appendChild(item);
  }

  // After the loop above, so every view has a recorded visibility to read.
  renderViewToggle(described.views);

  $("#dataset-summary").textContent = described.n_views
    ? `${described.n_views} · ${described.views[0].ndim}D`
    : "None";
  $("#empty-views").hidden = Boolean(described.n_views);
}

async function applyDescribed(described) {
  poolServeFailures = 0;
  reportedFailures.clear();
  state.session = described;
  // A view that is no longer loaded cannot stay selected: the selection names
  // source URLs, and a removed one would silently match nothing.
  const loaded = new Set(described.views.map((view) => view.url));
  for (const url of state.selectedViewUrls) {
    if (!loaded.has(url)) state.selectedViewUrls.delete(url);
  }
  state.previewRoute = null;
  state.previewMetadata = null;
  state.previewTransformKey = null;
  state.currentFusedLayerUrls.clear();
  renderViews(described);
  renderTransformKeys(described.transform_keys);
  renderChannelControls(described);
  renderPlacementScope(described);
  renderTimeSlider();
  renderDimensionFields(described);
  await refreshSessionSpec();
  await refreshViewer();
  setBusy(false);
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

/** Mount one folder and describe the OME-Zarr images it holds. */
async function collectSources(handle, requireSingleImage) {
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
      requireSingleImage
        ? `'${handle.name}' is not an OME-Zarr; when several folders are ` +
          "dropped at once each one must be a single OME-Zarr"
        : "no OME-Zarr found - drop either an OME-Zarr directory or a " +
          "folder containing several of them",
    );
  }

  if (requireSingleImage && !(images.length === 1 && images[0].path === "")) {
    throw new Error(
      `'${handle.name}' holds ${images.length} image(s); when several ` +
        "folders are dropped at once each one must itself be an OME-Zarr",
    );
  }

  return images.map((image) => ({
    url: `${API_BASE}/fs/${mount}${image.path ? "/" + image.path : ""}`,
    name: image.name,
  }));
}

/** Is this a file the CZI reader should be given? */
function isCziFile(handle) {
  return handle.kind === "file" && handle.name.toLowerCase().endsWith(".czi");
}

/**
 * Mount local files into every Python worker and return their directory.
 *
 * The session worker and the compute workers each mount the same `File`
 * objects: Python opens the file wherever the work happens, and a compute
 * worker rebuilding a session from source URLs alone would otherwise find
 * nothing at the path those URLs name.
 */
async function mountFiles(files) {
  const mount = crypto.randomUUID().slice(0, 8);

  const { path } = await sessionWorker.send({ type: "mount-files", mount, files });
  await Promise.all(
    pool.workers.map((channel) =>
      channel.send({ type: "mount-files", mount, files }),
    ),
  );

  state.fileMounts.push({ mount, files });
  return path;
}

/** Drop every mounted file from every worker. */
async function unmountAllFiles() {
  const mounts = state.fileMounts.splice(0);

  for (const { mount } of mounts) {
    await Promise.all(
      [sessionWorker, ...pool.workers].map((channel) =>
        channel.send({ type: "unmount-files", mount }).catch(() => {}),
      ),
    );
  }
}

/**
 * Open one or more dropped CZI files, each as a mosaic of tiles.
 *
 * Every tile of the file becomes a view, so one CZI is a whole dataset. How
 * many tiles that is only the reader knows, so the page mounts the file and
 * lets Python enumerate them.
 */
async function loadCziFiles(handles) {
  const files = await Promise.all(handles.map((handle) => handle.getFile()));
  const directory = await mountFiles(files);

  let append = Boolean(state.session && state.session.n_views);

  log(
    `mounted ${files.length} CZI file(s); reading tile positions` +
      (append ? " and adding to the loaded views" : ""),
  );
  setStatus("opening CZI", true);

  let described = null;
  for (const file of files) {
    described = await command("load_czi", {
      path: `${directory}/${file.name}`,
      name: file.name,
      replace: !append,
    });
    // Only the first file may replace; the rest extend what it loaded.
    append = true;
  }

  await applyDescribed(described);
  setStatus(`${described.n_views} view(s) loaded`);
  setBusy(false);
}

/**
 * Open one or more dropped folders.
 *
 * A single folder may be an OME-Zarr or a directory holding several of them.
 * When more than one is dropped the intent is unambiguous - each is one
 * image - so anything else is reported rather than guessed at.
 */
async function loadDirectories(handles) {
  const requireSingleImage = handles.length > 1;

  const sources = [];
  for (const handle of handles) {
    sources.push(...(await collectSources(handle, requireSingleImage)));
  }

  // Dropping more data adds tiles to the session instead of replacing it, so
  // a dataset can be assembled from several folders. "Clear" starts over.
  const append = Boolean(state.session && state.session.n_views);
  log(
    `found ${sources.length} OME-Zarr image(s) in ${handles.length} ` +
      `folder(s); ` +
      (append ? "adding to the loaded views" : "opening"),
  );
  setStatus("opening images", true);

  const described = await command("load", { sources, replace: !append });
  await applyDescribed(described);

  setStatus(`${described.n_views} view(s) loaded`);
  setBusy(false);
}

async function loadExample(name) {
  setStatus("generating the example dataset", true);

  // Presets are complete datasets. Replacing makes it possible to move
  // directly between 2D/3D and single/multichannel examples without first
  // clearing incompatible dimensions by hand.
  const described = await command("load_example", { name, replace: true });
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

  // Released only after Python has dropped its open handles, which `clear`
  // does. Unmounting under a live handle would fail reads rather than free
  // anything.
  await unmountAllFiles();

  // Everything derived from the old dataset goes, so that what is loaded next
  // starts from the same state a fresh page would.
  state.previewRoute = null;
  state.previewMetadata = null;
  state.previewTransformKey = null;
  state.previewVisibility = true;
  state.transformKey = null;
  state.layerSources = null;
  state.position = null;
  state.viewVisibility.clear();
  state.currentViewLayerUrls.clear();
  state.currentFusedLayerUrls.clear();
  state.channelVisibility.clear();
  state.placementChannels.clear();
  state.placementTimeRange = null;
  state.timeIndex = 0;
  state.timeVaryingTransforms = false;
  state.selectedViewUrls.clear();
  state.positionalColors = false;
  $("#positional-colors").checked = false;
  for (const timer of positionalColorTimers) clearTimeout(timer);
  positionalColorTimers = [];
  for (const timer of contrastSyncTimers) clearTimeout(timer);
  contrastSyncTimers = [];
  state.editableTransformKeys.clear();
  placementSignatures.clear();
  clearTimeout(placementSyncTimer);
  await applyDescribed(described);
  log("cleared all views");
  setStatus("drop a folder to begin");
}

function namedTransform(selector, fallback) {
  const value = $(selector).value.trim();
  if (!value) throw new Error("Enter a name for the new transform.");
  return value || fallback;
}

function dimensionValues(selector, { integers = false } = {}) {
  const values = {};
  for (const input of document.querySelectorAll(`${selector} input`)) {
    if (input.value.trim() === "") continue;
    const value = integers ? Number.parseInt(input.value, 10) : Number(input.value);
    if (!Number.isFinite(value) || value <= 0) {
      throw new Error(`${input.dataset.dimension} must be a positive number.`);
    }
    values[input.dataset.dimension] = value;
  }
  return Object.keys(values).length ? values : null;
}

async function doCreateTransform() {
  setBusy(true);
  setStatus("creating transform", true);
  try {
    const newTransformKey = namedTransform("#placement-transform-name", "manual");
    const result = await command("copy_transform", {
      source_transform_key: state.transformKey,
      new_transform_key: newTransformKey,
    });
    state.session.transform_keys = result.transform_keys;
    state.transformKey = result.transform_key;
    state.editableTransformKeys.add(result.transform_key);
    state.previewRoute = null;
    state.previewMetadata = null;
    state.previewTransformKey = null;
    renderViews(state.session);
    renderTransformKeys(result.transform_keys);
    await refreshSessionSpec();
    await refreshViewer();
    log(`created '${newTransformKey}' from '${result.source_transform_key}'`);
    setStatus(`editing ${newTransformKey}`);
  } finally {
    setBusy(false);
  }
}

async function doRegister() {
  setBusy(true);
  setStatus("registering", true);
  const started = performance.now();

  try {
    const result = await command("register", {
      options: {
        transform_key: state.transformKey,
        new_transform_key: namedTransform(
          "#registration-transform-name",
          "registered",
        ),
        reg_channel_index:
          $("#registration-channel").value === ""
            ? null
            : Number($("#registration-channel").value),
        registration_binning: dimensionValues("#registration-binning", {
          integers: true,
        }),
      },
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
    state.previewMetadata = null;
    state.previewTransformKey = null;
    renderViews(state.session);
    renderTransformKeys(result.transform_keys);
    await refreshSessionSpec();
    await refreshViewer();
    setStatus("registered");
  } finally {
    clearProgress();
    setBusy(false);
  }
}

async function doFusePreview() {
  setBusy(true);
  setStatus("preparing the fused preview", true);

  try {
    const options = fusionOptions();
    const result = await command("fuse_preview", {
      options,
    });
    state.previewRoute = result.route;
    state.previewMetadata = result.metadata;
    state.previewTransformKey = options.transform_key;
    state.previewVisibility = true;
    renderViews(state.session);
    await refreshSessionSpec();
    await refreshViewer();

    const shape = Object.entries(result.metadata.levels[0].shape)
      .map(([dim, size]) => `${dim}:${size}`)
      .join(" ");
    log(`fused preview ready (${shape}); chunks are computed on demand`);
    setStatus("fused preview added to the viewer");
  } finally {
    clearProgress();
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
        ...fusionOptions(),
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
    clearProgress();
    setBusy(false);
  }
}

function fusionOptions() {
  return {
    transform_key: state.transformKey,
    fusion_func: $("#fusion-method").value,
    blending_widths: dimensionValues("#blending-widths"),
    output_spacing: dimensionValues("#output-spacing"),
    output_chunksize: dimensionValues("#output-chunksizes", {
      integers: true,
    }),
  };
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
  // `build` covers the app's own sources as well as the wheel, so a change to
  // any worker script busts its cache too - index.html versions itself the
  // same way.
  const build = String(manifest.build || manifest.sha256 || "dev").slice(0, 12);
  state.build = build;

  const config = {
    ...state.config,
    wheel_url: new URL(
      `packages/${manifest.wheel}?v=${build}`,
      window.location.href,
    ).href,
    // A lockfile of our own, if the build produced one: it saves every worker
    // ~10 MB of matplotlib that nothing here imports. Pyodide's own lock is
    // used when it is absent, so an older deployment still boots.
    lock_url: manifest.pyodide_lock
      ? new URL(
          `packages/${manifest.pyodide_lock}?v=${build}`,
          window.location.href,
        ).href
      : null,
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
  // Not awaited yet. Every worker installs the same ~60 MB runtime, and the
  // service worker now collapses concurrent requests for a file into one
  // download, so booting them together costs one download instead of one per
  // worker - and their unpacking overlaps rather than queueing.
  const sessionBoot = sessionWorker.send({ type: "boot", config });

  const select = $("#worker-count");
  select.max = String(state.config.max_n_workers);
  select.value = String(Math.min(3, state.config.max_n_workers));
  select.disabled = false;
  select.addEventListener("change", () => {
    const requested = Math.max(
      0,
      Math.min(state.config.max_n_workers, Number(select.value) || 0),
    );
    select.value = String(requested);
    log(`compute workers: ${select.value}`);
    startWorkers();
  });

  // Boot them now rather than on the first action: a Pyodide runtime takes
  // seconds to start, and there is no reason for the user to wait for it.
  startWorkers();

  const info = (await sessionBoot).result;
  log(
    `python ${info.python} · numpy ${info.numpy} · zarr ${info.zarr} · ` +
      `dask ${info.dask} · multiview-stitcher ${info.multiview_stitcher} · ` +
      `build ${build}`,
  );
  $("#runtime-info").textContent =
    `Python ${info.python} · NumPy ${info.numpy} · Zarr ${info.zarr} · ` +
    `multiview-stitcher ${info.multiview_stitcher} · build ${build}`;

  claimTab();

  const { examples } = await command("examples", {});
  if (examples.length) {
    const menu = $("#example-menu");
    menu.replaceChildren();
    for (const example of examples) {
      const button = document.createElement("button");
      button.type = "button";
      button.role = "menuitem";
      button.dataset.load = "";
      button.dataset.example = example.name;
      button.textContent = example.label;
      menu.appendChild(button);
    }
    $("#example").disabled = false;
  }

  setStatus("drop a folder to begin");
  $("#dropzone").classList.remove("disabled");
}

function wireUi() {
  const dropzone = $("#dropzone");

  for (const [buttonSelector, dialogSelector] of [
    ["#log-button", "#log-dialog"],
    ["#viewer-help-button", "#viewer-help-dialog"],
    ["#about-button", "#about-dialog"],
  ]) {
    $(buttonSelector).addEventListener("click", () =>
      $(dialogSelector).showModal(),
    );
  }
  for (const button of document.querySelectorAll("[data-close-dialog]")) {
    button.addEventListener("click", () => button.closest("dialog").close());
  }
  for (const dialog of document.querySelectorAll("dialog")) {
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close();
    });
  }
  $("#clear-log").addEventListener("click", () => {
    $("#log").replaceChildren();
    $("#log-count").hidden = true;
  });

  for (const button of document.querySelectorAll(".main-tabs [data-tab]")) {
    button.addEventListener("click", () => {
      for (const peer of document.querySelectorAll(".main-tabs [data-tab]")) {
        const selected = peer === button;
        peer.setAttribute("aria-selected", String(selected));
        $(`#${peer.dataset.tab}-panel`).hidden = !selected;
      }
    });
  }
  for (const button of document.querySelectorAll(".sub-tabs [data-subtab]")) {
    button.addEventListener("click", () => {
      const tabs = button.closest(".sub-tabs");
      for (const peer of tabs.querySelectorAll("[data-subtab]")) {
        const selected = peer === button;
        peer.setAttribute("aria-selected", String(selected));
        $(`#${peer.dataset.subtab}`).hidden = !selected;
      }
    });
  }

  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));

  dropzone.addEventListener("click", (event) => {
    // Clicking the zone itself opens the folder picker; clicking either button
    // inside it means what the button says.
    if (event.target.closest("button")) return;
    $("#browse").click();
  });
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      $("#browse").click();
    }
  });

  dropzone.addEventListener("drop", async (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");

    const items = Array.from(event.dataTransfer.items || []);
    if (!items.length || !items[0].getAsFileSystemHandle) {
      log("this browser cannot read dropped folders; use the browse button", "error");
      return;
    }

    // Claim every handle before awaiting anything: a DataTransferItemList is
    // only valid for the duration of the event, so reading it after a yield
    // returns nothing.
    const claimed = items.map((item) => item.getAsFileSystemHandle());

    try {
      const handles = (await Promise.all(claimed)).filter(Boolean);
      const directories = handles.filter((handle) => handle.kind === "directory");
      const cziFiles = handles.filter(isCziFile);

      if (!directories.length && !cziFiles.length) {
        log("drop one or more folders, or CZI files", "error");
        return;
      }

      const ignored = handles.length - directories.length - cziFiles.length;
      if (ignored > 0) {
        log(
          `ignoring ${ignored} dropped file(s); only folders and .czi files ` +
            "are read",
          "warn",
        );
      }

      // A mixed drop is loaded folders-first, so that the tiles of a CZI
      // extend that set rather than the other way round - `load_czi` replaces
      // only when nothing is loaded yet.
      if (directories.length) await withPool(() => loadDirectories(directories));
      if (cziFiles.length) await withPool(() => loadCziFiles(cziFiles));
    } catch (error) {
      log(error.message, "error");
      setStatus(
        hasViews()
          ? "could not open that; the loaded views are unchanged"
          : "failed to open what was dropped",
      );
    }
  });

  $("#browse").addEventListener("click", async () => {
    try {
      const handle = await window.showDirectoryPicker({ mode: "read" });
      await withPool(() => loadDirectories([handle]));
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

  $("#browse-czi").addEventListener("click", async (event) => {
    // The dropzone opens the folder picker when clicked anywhere; this button
    // sits inside it and means something else.
    event.stopPropagation();
    try {
      const handles = await window.showOpenFilePicker({
        multiple: true,
        types: [
          {
            description: "Carl Zeiss Image files",
            accept: { "application/octet-stream": [".czi"] },
          },
        ],
      });
      await withPool(() => loadCziFiles(handles));
    } catch (error) {
      if (error.name === "AbortError") return;
      log(error.message, "error");
      setStatus(
        hasViews()
          ? "could not open that file; the loaded views are unchanged"
          : "failed to open the CZI file",
      );
    }
  });

  $("#time-slider").addEventListener("input", (event) => {
    const index = Number(event.target.value);
    state.timeIndex = index;
    $("#time-value").textContent = `${timeCoords()[index]} of ${
      timeCoords()[timeCoords().length - 1]
    }`;
    // The viewer's own position is the single source of truth for the
    // timepoint, so the slider moves that rather than keeping a copy: a scrub
    // in Neuroglancer and a drag here end up in the same place.
    viewer.setTimepoint(index);
  });

  $("#toggle-selection").addEventListener("click", toggleAllSelected);

  $("#toggle-views").addEventListener("click", () => {
    const views = state.session?.views || [];
    if (!views.length) return;
    const visible = !allViewsVisible(views);
    for (const view of views) state.viewVisibility.set(view.url, visible);
    renderViews(state.session);
    applyDisplayVisibility();
  });

  $("#transform-key").addEventListener("change", async (event) => {
    state.transformKey = event.target.value;
    log(`showing transform key '${state.transformKey}'`);
    // Before the refresh as well as after it: whether tiles may be placed by
    // hand is a property of the key, and the answer changes right now.
    syncManualPlacement();
    await refreshViewer();
  });

  for (const [selector, position] of [
    ["#placement-time-first", 0],
    ["#placement-time-last", 1],
  ]) {
    $(selector).addEventListener("input", (event) => {
      const range = [...(state.placementTimeRange ?? [0, 0])];
      range[position] = Number(event.target.value);
      // The two handles share a track, so the one being dragged pushes the
      // other rather than crossing it.
      if (range[0] > range[1]) range[1 - position] = range[position];
      state.placementTimeRange = range;
      $("#placement-time-first").value = String(range[0]);
      $("#placement-time-last").value = String(range[1]);
      updatePlacementTimeUi();
    });
  }

  $("#manual-placement").addEventListener("change", (event) => {
    syncManualPlacement();
    setStatus(
      event.target.checked
        ? `ctrl+drag a tile to place it in ${state.transformKey}`
        : "manual placement off",
    );
  });

  $("#positional-colors").addEventListener("change", async (event) => {
    state.positionalColors = event.target.checked;
    try {
      setStatus(
        state.positionalColors
          ? "computing positional colors"
          : "restoring channel colors",
        state.positionalColors,
      );
      await applyPositionalColors();
      setStatus(
        state.positionalColors
          ? "showing positional colors"
          : "showing channel colors",
      );
    } catch (error) {
      state.positionalColors = false;
      event.target.checked = false;
      viewer.setPositionalColors(null);
      log(`could not apply positional colors: ${error.message}`, "error");
      setStatus("could not apply positional colors");
    }
  });

  $("#example").addEventListener("click", () => {
    const menu = $("#example-menu");
    menu.hidden = !menu.hidden;
    $("#example").setAttribute("aria-expanded", String(!menu.hidden));
  });

  $("#example-menu").addEventListener("click", async (event) => {
    const button = event.target.closest("button[data-example]");
    if (!button) return;
    $("#example-menu").hidden = true;
    $("#example").setAttribute("aria-expanded", "false");
    try {
      await withPool(() => loadExample(button.dataset.example));
    } catch (error) {
      log(error.message, "error");
      setStatus("failed to load the example");
      setBusy(false);
    }
  });

  document.addEventListener("click", (event) => {
    if (event.target.closest(".example-picker")) return;
    $("#example-menu").hidden = true;
    $("#example").setAttribute("aria-expanded", "false");
  });

  $("#clear").addEventListener("click", async () => {
    try {
      await clearSession();
    } catch (error) {
      log(error.message, "error");
    }
  });

  for (const [action, handler] of Object.entries({
    "new-transform": doCreateTransform,
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

/**
 * Start the workers the current selection asks for, without blocking.
 *
 * Booting a Pyodide runtime takes seconds, so it happens as soon as the count
 * is known - at start-up, and whenever it changes - rather than on the first
 * action. Nothing awaits it here; the UI stays responsive and `withPool` only
 * waits if work arrives before the pool is up.
 */
function startWorkers() {
  const requested = Number($("#worker-count").value);
  if (!state.runtimeConfig || requested === pool.size) return;

  pool
    .resize(requested, state.runtimeConfig)
    .catch((error) => log(`could not start workers: ${error.message}`, "error"));
}

/** Make sure the compute pool matches the current selection, then act. */
async function withPool(action) {
  const requested = Number($("#worker-count").value);
  if (requested !== pool.size) {
    setStatus("starting compute workers", true);
    startWorkers();
  }
  await pool.ready();
  return await action();
}

wireUi();
boot().catch((error) => {
  log(error.message, "error");
  setStatus("failed to start");
});
