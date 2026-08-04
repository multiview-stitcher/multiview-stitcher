/**
 * Shared Pyodide bootstrap for the session worker and the compute workers.
 *
 * Both roles load the identical runtime - the same Pyodide build, the same
 * pinned dependencies and the same multiview-stitcher wheel - so that any
 * worker can rebuild the same Python state from a session spec. What differs
 * is only which entry point in `multiview_stitcher.browser.worker` they call.
 */

/* global loadPyodide */

let pyodide = null;
let api = null;

//: Boot is slow enough - tens of seconds, most of it downloading - that the
//: page has to be able to show how far along it is. Reporting the step number
//: rather than matching on the message keeps the two ends independent.
const BOOT_PHASES = 4;

/** Load Pyodide, the pinned dependencies and the multiview-stitcher wheel. */
async function bootRuntime(config, { log = () => {} } = {}) {
  if (api) return api;

  const phase = (message, step) =>
    log(message, { phase: step, phases: BOOT_PHASES });

  importScripts(`${config.pyodide_index_url}pyodide.js`);

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

/** Run a session-worker command; returns the parsed response. */
function callCommand(command, payload) {
  return JSON.parse(api.worker.handle_json(command, JSON.stringify(payload || {})));
}

/** Run a compute-worker task; returns the parsed response. */
function callTask(task) {
  return JSON.parse(api.worker.run_task_json(JSON.stringify(task)));
}

/**
 * Answer a virtual OME-Zarr request.
 *
 * `sessionSpec` is null in the session worker (which owns the live session)
 * and set in compute workers, which rebuild a read-only copy on demand.
 */
function callServe(route, key, sessionSpec) {
  // JSON, like every other call into Python. Handing over a live JS object
  // instead would convert its nulls to `JsNull` proxies rather than to None,
  // and those pass an `is not None` check and then fail deep inside numeric
  // code - far from the boundary that produced them.
  const spec = sessionSpec ? JSON.stringify(sessionSpec) : null;
  let result = null;

  try {
    result = api.worker.serve_route(route, key, spec);
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
