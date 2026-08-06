/**
 * The persistent worker that owns the dataset.
 *
 * Everything stateful - the opened views, the transform keys, the current
 * generation - lives here for the lifetime of the page, so images are opened
 * once and registration results accumulate in Python rather than being shipped
 * back and forth. Compute workers are stateless by comparison and rebuild what
 * they need from a spec this worker hands them.
 */

// Started, not awaited. `self.onmessage` below has to be installed
// before this module's first suspension point: a message that arrives
// while a top-level `await` is pending has no handler to reach, and the
// worker then sits there having missed the one message it was sent.
//
// The build id is carried on so the shared runtime is not loaded from
// cache while the worker itself is fresh.
const runtime = import(`./py-runtime.js${self.location.search}`);

let ready = false;
let outputMount = null;

const OUTPUT_PATH = "/output";

function post(id, payload) {
  self.postMessage({ id, ...payload });
}

self.onmessage = async (event) => {
  const { bootRuntime, callCommand, callServe } = await runtime;
  const { id, type } = event.data;

  try {
    if (type === "boot") {
      await bootRuntime(event.data.config, {
        log: (message, progress) =>
          self.postMessage({ type: "log", message, ...progress }),
      });
      ready = true;
      post(id, { ok: true, result: (await callCommand("info", {})).result });
      return;
    }

    if (!ready) throw new Error("the Python runtime is still starting");

    if (type === "command") {
      const response = await callCommand(event.data.command, event.data.payload);
      if (!response.ok) throw new Error(response.error + "\n" + (response.traceback || ""));
      post(id, { ok: true, result: response.result });
      return;
    }

    if (type === "serve") {
      const response = await callServe(event.data.route, event.data.key, null);
      post(id, response, response.found ? [response.data] : []);
      return;
    }

    throw new Error(`unknown session-worker message '${type}'`);
  } catch (error) {
    post(id, { error: String((error && error.message) || error) });
  }
};
