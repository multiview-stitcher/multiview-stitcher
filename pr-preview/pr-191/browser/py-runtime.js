/**
 * Shared Pyodide bootstrap for the session worker and the compute workers.
 *
 * Both roles load the identical runtime - the same Pyodide build, the same
 * pinned dependencies and the same multiview-stitcher wheel - so that any
 * worker can rebuild the same Python state from a session spec. What differs
 * is only which entry point in `multiview_stitcher.browser.worker` they call.
 *
 * An ES module, and so are the workers that load it: Pyodide refuses to start
 * in a classic worker ("Classic web workers are not supported"), so the whole
 * chain - `new Worker(..., {type: "module"})`, `import` rather than
 * `importScripts` - has to be modules too.
 */

export let pyodide = null;
let api = null;

//: Boot is slow enough - tens of seconds, most of it downloading - that the
//: page has to be able to show how far along it is. Reporting the step number
//: rather than matching on the message keeps the two ends independent.
const BOOT_PHASES = 4;

/** Load Pyodide, the pinned dependencies and the multiview-stitcher wheel. */
export async function bootRuntime(config, { log = () => {} } = {}) {
  if (api) return api;

  const phase = (message, step) =>
    log(message, { phase: step, phases: BOOT_PHASES });

  // Imported rather than `importScripts`-ed: this is a module worker, and
  // Pyodide's own module build is what it offers for one.
  const { loadPyodide } = await import(
    /* webpackIgnore: true */ `${config.pyodide_index_url}pyodide.mjs`
  );

  phase("booting Python runtime", 1);
  pyodide = await loadPyodide({
    indexURL: config.pyodide_index_url,
    // Our lockfile only differs from Pyodide's in the dependency graph; see
    // `write_pyodide_lock` in scripts/build_browser_app.py.
    //
    // `packageBaseUrl` has to be given with it. Pyodide otherwise defaults it
    // to the directory the lockfile came from, and would look for every wheel
    // next to our copy rather than in the distribution - where the files do
    // not exist, and whose checksums could not match if they did.
    ...(config.lock_url
      ? {
          lockFileURL: config.lock_url,
          packageBaseUrl: config.pyodide_index_url,
        }
      : {}),
    packages: config.pyodide_packages,
  });

  phase("installing dependencies", 2);
  await pyodide.runPythonAsync(`
import micropip
await micropip.install(${JSON.stringify(config.browser_dependencies)})
`);

  phase("installing multiview-stitcher", 3);
  await pyodide.runPythonAsync(`
import micropip
await micropip.install(${JSON.stringify(config.wheel_url)}, deps=False)
`);

  // The bridge lets synchronous Python block on work done by the pool. Only a
  // Web Worker may issue the synchronous requests it relies on, which is
  // exactly why every Python role here lives in a worker.
  await pyodide.runPythonAsync(`
from multiview_stitcher.browser.bridge import XHRBridge, set_bridge
set_bridge(XHRBridge(base_url=${JSON.stringify(config.api_base)}))
`);

  const worker = pyodide.pyimport("multiview_stitcher.browser.worker");
  api = { pyodide, worker };

  phase("ready", 4);
  return api;
}

/**
 * Where mounted CZI files appear in this worker's Python filesystem: one
 * directory per mount id, so two files of the same name dropped from different
 * folders cannot collide.
 */
const CZI_ROOT = "/czi";

//: Mount id -> path, for mounts this worker already holds. Every worker mounts
//: the same files independently - Python opens the file wherever it runs - and
//: the page replays mounts to workers started later, so an id arrives twice.
const cziMounts = new Map();

/**
 * Mount local files so Python can open them by path.
 *
 * WORKERFS is what makes reading a multi-gigabyte CZI possible at all: it
 * serves reads straight from the `File` through `Blob.slice` and
 * `FileReaderSync`, so seeking around the file costs only the bytes actually
 * read and the file is never copied into the WebAssembly heap. It is
 * worker-only - `FileReaderSync` does not exist on the main thread - which is
 * one more reason every Python role in this app lives in a worker.
 *
 * Returns the directory the files were mounted at.
 */
export function mountFiles(mountId, files) {
  if (cziMounts.has(mountId)) return cziMounts.get(mountId);

  const { FS } = pyodide;
  const path = `${CZI_ROOT}/${mountId}`;

  FS.mkdirTree(path);
  FS.mount(FS.filesystems.WORKERFS, { files }, path);
  cziMounts.set(mountId, path);

  return path;
}

/** Release a mount so the browser can let go of the underlying file. */
export function unmountFiles(mountId) {
  if (!cziMounts.has(mountId)) return;

  const { FS } = pyodide;
  const path = cziMounts.get(mountId);

  FS.unmount(path);
  FS.rmdir(path);
  cziMounts.delete(mountId);
}

// Every call into Python is made with `callPromising`, which lets the
// WebAssembly stack suspend. zarr v3 needs it: its API is asynchronous
// underneath, and with no thread in the browser to run an event loop on it
// blocks by stack switching instead. A plain synchronous call fails with
// "Cannot stack switch because the Python entrypoint was a synchronous
// function", so the requirement announces itself rather than hiding.
//
// Suspending returns control to the JavaScript event loop mid-call, so a
// second message could otherwise start while the first is still inside
// Python. The session is stateful and its Python is not written to be
// re-entered, so calls are serialised: one at a time, in arrival order.
let pythonTurn = Promise.resolve();

function inTurn(work) {
  const turn = pythonTurn.then(work, work);
  // The queue must neither stall on a failure nor report one twice.
  pythonTurn = turn.then(
    () => {},
    () => {},
  );
  return turn;
}

/** Run a session-worker command; returns the parsed response. */
export function callCommand(command, payload) {
  return inTurn(async () =>
    JSON.parse(
      await api.worker.handle_json.callPromising(
        command,
        JSON.stringify(payload || {}),
      ),
    ),
  );
}

/** Run a compute-worker task; returns the parsed response. */
export function callTask(task) {
  return inTurn(async () =>
    JSON.parse(await api.worker.run_task_json.callPromising(JSON.stringify(task))),
  );
}

/**
 * Answer a virtual OME-Zarr request.
 *
 * `sessionSpec` is null in the session worker (which owns the live session)
 * and set in compute workers, which rebuild a read-only copy on demand.
 */
export function callServe(route, key, sessionSpec) {
  return inTurn(() => serveOnce(route, key, sessionSpec));
}

async function serveOnce(route, key, sessionSpec) {
  // JSON, like every other call into Python. Handing over a live JS object
  // instead would convert its nulls to `JsNull` proxies rather than to None,
  // and those pass an `is not None` check and then fail deep inside numeric
  // code - far from the boundary that produced them.
  const spec = sessionSpec ? JSON.stringify(sessionSpec) : null;
  let result = null;

  try {
    result = await api.worker.serve_route.callPromising(route, key, spec);
    // (status, content type, body) with the body as bytes, which `toJs`
    // converts to a Uint8Array view on the WebAssembly heap.
    const [status, contentType, body] = result.toJs();

    if (status >= 500) {
      // Carries the Python traceback; surfacing it is the difference between
      // a debuggable failure and a silently empty layer.
      throw new Error(new TextDecoder().decode(body));
    }
    if (status !== 200 || !body) {
      // A 404 explains itself. Some are routine - the viewer probes for keys
      // that legitimately do not exist - so the page decides what is worth
      // reporting rather than treating every one as a failure.
      return {
        found: false,
        reason: body ? new TextDecoder().decode(body) : "not found",
      };
    }

    // Copy out of the heap: the view is invalidated as soon as Python frees
    // the object, and the buffer is transferred to another thread from here.
    return {
      found: true,
      data: new Uint8Array(body).slice().buffer,
      contentType,
    };
  } finally {
    if (result) result.destroy();
  }
}
