/**
 * multiview-stitcher service worker.
 *
 * Turns three kinds of same-origin HTTP request into work done elsewhere in
 * the app, which is what lets Neuroglancer and synchronous Python code share
 * one addressing scheme:
 *
 *   GET  <base>/__mvs__/fs/<mount>/<path>     -> a read from a local directory
 *                                               the user granted access to
 *   PUT/DELETE  same                          -> a write to, or removal from,
 *                                               that directory
 *   GET  <base>/__mvs__/zarr/<route>/<key>    -> a chunk computed in Python
 *   POST <base>/__mvs__/rpc/dispatch          -> start work on the pool
 *   POST <base>/__mvs__/rpc/poll              -> ask how that work is going
 *
 * The worker holds no state of its own: every request is forwarded to the
 * page, which owns the directory handles and the worker pool. Requests are
 * answered on a MessageChannel port so that many can be in flight at once.
 */

/* global mvsRoutes */

importScripts("routes.js");

// A file read is quick and should fail loudly rather than hang. Computing a
// chunk in Python legitimately takes far longer, especially on the first
// request to a worker, which opens the inputs.
//
// An RPC no longer waits for the work it starts: the page runs the job and
// answers a poll within seconds, whatever the job goes on to do. So this
// bounds how long the *page* may take to reply, not how long a registration
// runs - and a browser terminating this worker mid-request is no longer a way
// to lose one.
const FILE_TIMEOUT_MS = 60 * 1000;
const COMPUTE_TIMEOUT_MS = 10 * 60 * 1000;
const RPC_TIMEOUT_MS = 2 * 60 * 1000;

self.addEventListener("install", (event) => {
  // Take over immediately: the page cannot read anything until we are active.
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      await self.clients.claim();
      // Drop caches from an older layout rather than leaving tens of
      // megabytes of unreachable entries behind.
      for (const name of await caches.keys()) {
        if (name !== RUNTIME_CACHE) await caches.delete(name);
      }
    })(),
  );
});

// ---------------------------------------------------------------------------
// The Python runtime cache
// ---------------------------------------------------------------------------

/**
 * Pyodide, its packages and the wheels micropip pulls from PyPI.
 *
 * Every worker installs an identical runtime - around 60 MB - and they boot at
 * the same time, so without this they all miss the HTTP cache together and the
 * same bytes are fetched once per worker. Serving them from here makes the
 * network see each file once ever: concurrent requests for a file that is on
 * its way are joined to the one in flight, and a later visit is served from
 * disk.
 *
 * These URLs all carry their version in the path, so an entry can never go
 * stale - a different Pyodide or a different wheel is simply a different URL.
 */
const RUNTIME_CACHE = "mvs-python-runtime-v1";

const RUNTIME_ASSET = /\.(whl|wasm|zip|data|js|mjs|json)$/;

/** Fetches in progress, so that N workers wanting one file make one request. */
const inFlight = new Map();

async function serveRuntimeAsset(request) {
  const url = request.url;
  const cache = await caches.open(RUNTIME_CACHE);

  const cached = await cache.match(url);
  if (cached) return cached;

  let pending = inFlight.get(url);
  if (!pending) {
    pending = (async () => {
      const response = await fetch(request);
      // An opaque response has no readable body to hand on to several
      // callers, and storing one costs far more quota than it is worth.
      if (!response.ok || (response.type !== "cors" && response.type !== "basic")) {
        return false;
      }
      await cache.put(url, response.clone());
      return true;
    })();
    pending.finally(() => inFlight.delete(url));
    inFlight.set(url, pending);
  }

  // Quota errors and network failures are not worth failing a boot over.
  const stored = await pending.catch(() => false);
  if (stored) {
    const hit = await cache.match(url);
    if (hit) return hit;
  }

  return fetch(request);
}

function clientRank(client) {
  if (client.focused) return 0;
  if (client.visibilityState === "visible") return 1;
  return 2;
}

function askClient(client, message, timeoutMs) {
  return new Promise((resolve, reject) => {
    const channel = new MessageChannel();
    const timer = setTimeout(() => {
      reject(new Error(`timed out after ${timeoutMs} ms`));
    }, timeoutMs);

    channel.port1.onmessage = (event) => {
      clearTimeout(timer);
      const data = event.data || {};
      if (data.error) reject(new Error(data.error));
      else resolve(data);
    };

    client.postMessage({ ...message, port: channel.port2 }, [channel.port2]);
  });
}

/**
 * Ask a page to answer one request.
 *
 * Two things make this less obvious than it looks. The embedded Neuroglancer
 * viewer is an iframe, and an iframe is a *window* client just like the page
 * is - but it has no handler for our messages, so asking it means waiting for
 * a reply that never comes; since the viewer issues most of the requests we
 * serve, that deadlocks exactly when the app is busiest. And one service
 * worker serves every tab, each of which owns its own directory handles and
 * session, so a tab that does not recognise a mount or session replies
 * `notMine` and the request moves on to the next one.
 */
async function askPage(message, timeoutMs = FILE_TIMEOUT_MS) {
  const clients = (
    await self.clients.matchAll({ type: "window", includeUncontrolled: true })
  )
    .filter((client) =>
      mvsRoutes.isAppPage(client.url, self.registration.scope),
    )
    // Ask the tab the user is actually looking at first. Older tabs may not
    // decline requests they cannot serve, and whichever answers first wins -
    // so preferring the visible one keeps a forgotten background tab from
    // answering for the foreground one.
    .sort((a, b) => clientRank(a) - clientRank(b));

  if (!clients.length) {
    throw new Error("no multiview-stitcher page is available to serve this");
  }

  let lastError = null;
  for (const client of clients) {
    try {
      const response = await askClient(client, message, timeoutMs);
      if (!response.notMine) return response;
    } catch (error) {
      lastError = error;
    }
  }

  // Never report this as "not found": zarr treats a missing chunk as a hole
  // and quietly fills it with zeros, which turns a broken request path into a
  // black image with nothing in any log.
  throw (
    lastError ||
    new Error(`no page claimed this request (${message.type})`)
  );
}

const NO_STORE = {
  // Never let the HTTP cache hold on to computed chunks: routes carry a
  // session generation and are retired the moment the data behind them
  // changes, so a cached response could only ever be a stale one.
  "Cache-Control": "no-store",
  "Access-Control-Allow-Origin": "*",
};

function notFound(detail) {
  return new Response(detail || "not found", {
    status: 404,
    headers: {
      ...NO_STORE,
      // Repeated as a header: the browser's network panel shows headers at a
      // glance, while a 404 body usually has to be hunted for.
      "X-Mvs-Reason": String(detail || "not found").slice(0, 200),
    },
  });
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

  if (request.method === "PUT" || request.method === "DELETE") {
    // One request writes or removes exactly one file, so many can be in
    // flight at once as long as they name different files.
    const body =
      request.method === "PUT" ? await request.arrayBuffer() : null;
    await askPage(
      {
        type: "fs.write",
        mount: parsed.mount,
        path: parsed.path,
        data: body,
      },
      FILE_TIMEOUT_MS,
    );
    return new Response(null, { status: 204, headers: NO_STORE });
  }

  const response = await askPage({
    type: "fs.read",
    mount: parsed.mount,
    path: parsed.path,
  });
  if (!response.found) return notFound(`no file at ${path}`);

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

  const response = await askPage(
    { type: "zarr.read", route: parsed.route, key: parsed.key },
    COMPUTE_TIMEOUT_MS,
  );
  if (!response.found) return notFound(response.reason);

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
  const response = await askPage(
    { type: "rpc", endpoint, payload },
    RPC_TIMEOUT_MS,
  );
  return new Response(JSON.stringify(response.result), {
    status: 200,
    headers: { ...NO_STORE, "Content-Type": "application/json" },
  });
}

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // The Python runtime comes from a CDN and from PyPI; everything else
  // cross-origin is left alone.
  if (
    url.origin !== self.location.origin &&
    event.request.method === "GET" &&
    RUNTIME_ASSET.test(url.pathname)
  ) {
    event.respondWith(serveRuntimeAsset(event.request));
    return;
  }

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
