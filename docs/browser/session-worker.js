/**
 * The persistent worker that owns the dataset.
 *
 * Everything stateful - the opened views, the transform keys, the current
 * generation - lives here for the lifetime of the page, so images are opened
 * once and registration results accumulate in Python rather than being shipped
 * back and forth. Compute workers are stateless by comparison and rebuild what
 * they need from a spec this worker hands them.
 */

/* global bootRuntime, callCommand, callServe, pyodide */

// Carry the build id on, so the shared runtime is not loaded from cache
// while the worker itself is fresh.
importScripts(`py-runtime.js${self.location.search}`);

let ready = false;
let outputMount = null;

const OUTPUT_PATH = "/output";

function post(id, payload) {
  self.postMessage({ id, ...payload });
}

self.onmessage = async (event) => {
  const { id, type } = event.data;

  try {
    if (type === "boot") {
      await bootRuntime(event.data.config, {
        log: (message) => self.postMessage({ type: "log", message }),
      });
      ready = true;
      post(id, { ok: true, result: callCommand("info", {}).result });
      return;
    }

    if (!ready) throw new Error("the Python runtime is still starting");

    if (type === "command") {
      const response = callCommand(event.data.command, event.data.payload);
      if (!response.ok) throw new Error(response.error + "\n" + (response.traceback || ""));
      post(id, { ok: true, result: response.result });
      return;
    }

    if (type === "serve") {
      const response = callServe(event.data.route, event.data.key, null);
      post(id, response, response.found ? [response.data] : []);
      return;
    }

    // Fusing to disk needs a writable filesystem. The chosen output directory
    // is mounted into this worker's Emscripten filesystem, so Python writes
    // ordinary paths; `sync_output` flushes them back to the real directory.
    // Only this worker mounts it, which keeps the writes serialised - unlike
    // reads, concurrent mounts of one directory are not safe to reconcile.
    if (type === "mount_output") {
      if (outputMount) {
        await outputMount.syncfs();
        pyodide.FS.unmount(OUTPUT_PATH);
        outputMount = null;
      }
      try {
        pyodide.FS.mkdirTree(OUTPUT_PATH);
      } catch (error) {
        /* already exists */
      }
      outputMount = await pyodide.mountNativeFS(OUTPUT_PATH, event.data.handle);
      post(id, { ok: true });
      return;
    }

    if (type === "sync_output") {
      if (!outputMount) throw new Error("no output directory is mounted");
      await outputMount.syncfs();
      post(id, { ok: true });
      return;
    }

    throw new Error(`unknown session-worker message '${type}'`);
  } catch (error) {
    post(id, { error: String((error && error.message) || error) });
  }
};
