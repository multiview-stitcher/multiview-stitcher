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

/** Load Pyodide, the pinned dependencies and the multiview-stitcher wheel. */
async function bootRuntime(config, { log = () => {} } = {}) {
  if (api) return api;

  importScripts(`${config.pyodide_index_url}pyodide.js`);

  log("booting Python runtime");
  pyodide = await loadPyodide({
    indexURL: config.pyodide_index_url,
    packages: config.pyodide_packages,
  });

  log("installing dependencies");
  await pyodide.runPythonAsync(`
import micropip
await micropip.install(${JSON.stringify(config.browser_dependencies)})
`);

  log("installing multiview-stitcher");
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

  log("ready");
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
  const spec = sessionSpec ? api.pyodide.toPy(sessionSpec) : null;
  let result = null;

  try {
    result = api.worker.serve_route(route, key, spec);
    // (status, content type, body) with the body as bytes, which `toJs`
    // converts to a Uint8Array view on the WebAssembly heap.
    const [status, contentType, body] = result.toJs();

    if (status !== 200 || !body) return { found: false };

    // Copy out of the heap: the view is invalidated as soon as Python frees
    // the object, and the buffer is transferred to another thread from here.
    return {
      found: true,
      data: new Uint8Array(body).slice().buffer,
      contentType,
    };
  } finally {
    if (result) result.destroy();
    if (spec) spec.destroy();
  }
}
