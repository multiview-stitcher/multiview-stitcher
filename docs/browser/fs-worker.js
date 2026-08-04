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

// Directory handles are cached so that reading a chunk does not re-walk the
// path every time. Bounded because a deep pyramid has a directory per chunk
// row, and an unbounded cache would grow with the dataset rather than with
// what is being looked at.
const MAX_CACHED_DIRECTORIES = 4096;

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

  if (mount.dirCache.size >= MAX_CACHED_DIRECTORIES) {
    mount.dirCache.delete(mount.dirCache.keys().next().value);
  }
  mount.dirCache.set(key, handle);
  return handle;
}

/** The id of an already-mounted directory identical to `handle`, if any. */
async function findMount(handle) {
  for (const [id, mount] of mounts) {
    try {
      if (await mount.handle.isSameEntry(handle)) return id;
    } catch (error) {
      /* handles from a previous session may no longer compare */
    }
  }
  return null;
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
    // "No file here" is a normal answer, not a failure: zarr probes for keys
    // that may not exist and reads an absent chunk as its fill value. A path
    // that names a directory (TypeMismatchError) means the same thing - and
    // reporting it as an error would abort a whole read in Python, which
    // fetches strictly, even though Neuroglancer would shrug it off.
    if (error && (error.name === "NotFoundError" || error.name === "TypeMismatchError")) {
      return null;
    }
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
  // The transfer list matters: chunk buffers are handed over rather than
  // copied, which halves the memory traffic of reading a large dataset.
  const reply = (payload, transfer = []) =>
    port
      ? port.postMessage(payload, transfer)
      : self.postMessage({ id, ...payload }, transfer);

  try {
    if (type === "mount") {
      // Dropping the same folder twice must address the same mount, otherwise
      // its images would be appended again under new URLs and appear as
      // duplicate views.
      const existing = await findMount(event.data.handle);
      if (existing) {
        reply({ ok: true, mount: existing });
        return;
      }

      mounts.set(event.data.mount, {
        handle: event.data.handle,
        dirCache: new Map(),
      });
      reply({ ok: true, mount: event.data.mount });
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
