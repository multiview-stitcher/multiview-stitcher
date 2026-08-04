/**
 * multiview-stitcher service worker.
 *
 * Turns three kinds of same-origin HTTP request into work done elsewhere in
 * the app, which is what lets Neuroglancer and synchronous Python code share
 * one addressing scheme:
 *
 *   GET  <base>/__mvs__/fs/<mount>/<path>     -> a read from a local directory
 *                                               the user granted access to
 *   GET  <base>/__mvs__/zarr/<route>/<key>    -> a chunk computed in Python
 *   POST <base>/__mvs__/rpc/<endpoint>        -> work farmed out to the pool
 *
 * The worker holds no state of its own: every request is forwarded to the
 * page, which owns the directory handles and the worker pool. Requests are
 * answered on a MessageChannel port so that many can be in flight at once.
 */

/* global mvsRoutes */

importScripts("routes.js");

const TIMEOUT_MS = 10 * 60 * 1000;

self.addEventListener("install", (event) => {
  // Take over immediately: the page cannot read anything until we are active.
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

/** Ask the controlling page to answer one request. */
async function askPage(message, transfer = []) {
  const clients = await self.clients.matchAll({
    type: "window",
    includeUncontrolled: true,
  });

  if (!clients.length) {
    throw new Error("no page is available to serve this request");
  }

  return await new Promise((resolve, reject) => {
    const channel = new MessageChannel();
    const timer = setTimeout(() => {
      reject(new Error(`timed out after ${TIMEOUT_MS} ms`));
    }, TIMEOUT_MS);

    channel.port1.onmessage = (event) => {
      clearTimeout(timer);
      const data = event.data || {};
      if (data.error) reject(new Error(data.error));
      else resolve(data);
    };

    // Prefer the client that triggered this request; fall back to any window.
    clients[0].postMessage({ ...message, port: channel.port2 }, [
      channel.port2,
      ...transfer,
    ]);
  });
}

const NO_STORE = {
  // Never let the HTTP cache hold on to computed chunks: routes carry a
  // session generation and are retired the moment the data behind them
  // changes, so a cached response could only ever be a stale one.
  "Cache-Control": "no-store",
  "Access-Control-Allow-Origin": "*",
};

function notFound(detail) {
  return new Response(detail || "not found", { status: 404, headers: NO_STORE });
}

function serverError(error) {
  return new Response(String(error && error.message ? error.message : error), {
    status: 500,
    headers: NO_STORE,
  });
}

async function handleFile(path, request) {
  const parsed = mvsRoutes.parseFilePath(path);
  if (!parsed) return notFound("missing mount id");

  const response = await askPage({
    type: "fs.read",
    mount: parsed.mount,
    path: parsed.path,
  });
  if (!response.found) return notFound();

  return new Response(request.method === "HEAD" ? null : response.data, {
    status: 200,
    headers: {
      ...NO_STORE,
      "Content-Type": "application/octet-stream",
      "Content-Length": String(response.data.byteLength),
    },
  });
}

async function handleZarr(path, request) {
  const parsed = mvsRoutes.parseZarrPath(path);
  if (!parsed) return notFound("malformed virtual OME-Zarr route");

  const response = await askPage({
    type: "zarr.read",
    route: parsed.route,
    key: parsed.key,
  });
  if (!response.found) return notFound();

  return new Response(request.method === "HEAD" ? null : response.data, {
    status: 200,
    headers: {
      ...NO_STORE,
      "Content-Type": response.contentType || "application/octet-stream",
      "Content-Length": String(
        response.data.byteLength !== undefined
          ? response.data.byteLength
          : response.data.length,
      ),
    },
  });
}

async function handleRpc(endpoint, request) {
  const payload = await request.json();
  const response = await askPage({ type: "rpc", endpoint, payload });
  return new Response(JSON.stringify(response.result), {
    status: 200,
    headers: { ...NO_STORE, "Content-Type": "application/json" },
  });
}

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return;

  const parsed = mvsRoutes.parseRequest(url.pathname);
  if (!parsed) return;

  const { kind, path } = parsed;

  event.respondWith(
    (async () => {
      try {
        if (kind === "fs") return await handleFile(path, event.request);
        if (kind === "zarr") return await handleZarr(path, event.request);
        if (kind === "rpc") return await handleRpc(path, event.request);
        if (kind === "ping") {
          return new Response("ok", { status: 200, headers: NO_STORE });
        }
        return notFound(`unknown route kind '${kind}'`);
      } catch (error) {
        return serverError(error);
      }
    })(),
  );
});
