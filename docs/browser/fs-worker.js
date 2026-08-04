/**
 * Owns the File System Access API handles for directories the user granted.
 *
 * Reads happen here rather than on the page so that the main thread stays free
 * while Python workers block on synchronous requests for chunks. Directory
 * handles are structured-cloneable, so the page transfers them here once and
 * never touches file contents itself.
 */

/** mount id -> { handle, dirCache } */
const mounts = new Map();

function mountFor(id) {
  const mount = mounts.get(id);
  if (!mount) throw new Error(`unknown mount '${id}'`);
  return mount;
}

async function directoryAt(mount, segments) {
  const key = segments.join("/");
  if (mount.dirCache.has(key)) return mount.dirCache.get(key);

  let handle = mount.handle;
  for (const segment of segments) {
    handle = await handle.getDirectoryHandle(segment);
  }

  mount.dirCache.set(key, handle);
  return handle;
}

async function readFile(id, path) {
  const mount = mountFor(id);
  const segments = path.split("/").filter(Boolean);
  if (!segments.length) return null;

  const name = segments.pop();

  try {
    const directory = await directoryAt(mount, segments);
    const fileHandle = await directory.getFileHandle(name);
    const file = await fileHandle.getFile();
    return await file.arrayBuffer();
  } catch (error) {
    // NotFoundError is the normal "this chunk was never written" answer that
    // zarr relies on; anything else is worth surfacing.
    if (error && error.name === "NotFoundError") return null;
    throw error;
  }
}

/** Does this directory look like the root of an OME-Zarr image? */
async function isOmeZarr(handle) {
  for (const name of [".zattrs", "zarr.json"]) {
    try {
      const file = await (await handle.getFileHandle(name)).getFile();
      const attrs = JSON.parse(await file.text());
      if (attrs.multiscales) return true;
      if (attrs.ome && attrs.ome.multiscales) return true;
      if (attrs.attributes && attrs.attributes.ome) return true;
    } catch (error) {
      if (!error || error.name !== "NotFoundError") throw error;
    }
  }
  return false;
}

/**
 * Find the OME-Zarr images inside a granted directory.
 *
 * Accepts either a directory that *is* an OME-Zarr, or one that contains
 * several of them as immediate children - which is how tiles usually arrive.
 */
async function discover(id) {
  const mount = mountFor(id);

  if (await isOmeZarr(mount.handle)) {
    return [{ name: mount.handle.name, path: "" }];
  }

  const found = [];
  for await (const [name, handle] of mount.handle.entries()) {
    if (handle.kind !== "directory") continue;
    if (await isOmeZarr(handle)) found.push({ name, path: name });
  }

  found.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  return found;
}

self.onmessage = async (event) => {
  const { id, type, port } = event.data;
  const reply = (payload) => (port ? port.postMessage(payload) : self.postMessage({ id, ...payload }));

  try {
    if (type === "mount") {
      mounts.set(event.data.mount, {
        handle: event.data.handle,
        dirCache: new Map(),
      });
      reply({ ok: true });
      return;
    }

    if (type === "discover") {
      reply({ ok: true, images: await discover(event.data.mount) });
      return;
    }

    if (type === "read") {
      const data = await readFile(event.data.mount, event.data.path);
      if (data === null) {
        reply({ found: false });
      } else {
        reply({ found: true, data }, [data]);
      }
      return;
    }

    if (type === "unmount") {
      mounts.delete(event.data.mount);
      reply({ ok: true });
      return;
    }

    reply({ error: `unknown fs-worker message '${type}'` });
  } catch (error) {
    reply({ error: String((error && error.message) || error) });
  }
};
