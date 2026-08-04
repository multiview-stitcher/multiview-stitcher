/**
 * A stateless compute worker.
 *
 * Runs pairwise registrations, fusion blocks and virtual OME-Zarr chunk
 * requests. Each task carries a session spec - source URLs plus the current
 * transforms - from which the worker rebuilds the same Python objects the
 * session worker holds, caching them per session generation. Image data is
 * read straight from the granted directory through the service worker, so it
 * never travels through JavaScript.
 */

/* global bootRuntime, callTask, callServe */

importScripts("py-runtime.js");

let ready = false;

function post(id, payload, transfer = []) {
  self.postMessage({ id, ...payload }, transfer);
}

self.onmessage = async (event) => {
  const { id, type } = event.data;

  try {
    if (type === "boot") {
      await bootRuntime(event.data.config, {
        log: (message) =>
          self.postMessage({ type: "log", message, worker: event.data.name }),
      });
      ready = true;
      post(id, { ok: true });
      return;
    }

    if (!ready) throw new Error("the Python runtime is still starting");

    if (type === "task") {
      const response = callTask(event.data.task);
      if (!response.ok) {
        post(id, { ok: true, result: { error: response.error } });
      } else {
        post(id, { ok: true, result: response.result });
      }
      return;
    }

    if (type === "serve") {
      const response = callServe(
        event.data.route,
        event.data.key,
        event.data.session,
      );
      post(id, response, response.found ? [response.data] : []);
      return;
    }

    throw new Error(`unknown compute-worker message '${type}'`);
  } catch (error) {
    post(id, { error: String((error && error.message) || error) });
  }
};
