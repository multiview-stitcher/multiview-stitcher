"""
zarr stores backed by HTTP.

In the browser, OME-Zarr inputs live in a directory the user granted access to
through the File System Access API. Python never touches those files directly:
a same-origin service worker exposes them under stable URLs and translates each
request into a filesystem read. This module turns such a URL prefix into a zarr
store, so `zarr.open_group(store=...)` works unchanged and Neuroglancer and
Python read exactly the same bytes.

The fetch backend is pluggable, which keeps the store testable on CPython:

* in a Pyodide **worker**, a synchronous ``XMLHttpRequest`` is used - workers
  may block, and this is what lets ordinary synchronous zarr/dask code drive
  asynchronous browser IO without SharedArrayBuffer or special COOP/COEP
  headers (which GitHub Pages cannot set);
* on CPython, ``urllib`` is used, or any callable the caller supplies.

Writing works the same way, in reverse: a ``PUT`` reaches a worker holding the
output directory handle, which writes exactly one file and closes it. Because
each zarr chunk is its own file, any number of workers can write concurrently
as long as no two touch the same key - which is what makes fusing to disk
parallel without a shared flush step.
"""

import urllib.error
import urllib.request
from collections.abc import MutableMapping

import zarr

from multiview_stitcher._zarr_compat import ZARR_V3
from multiview_stitcher.browser.env import is_pyodide

#: Keys whose contents are small, immutable per generation, and requested
#: repeatedly by zarr; caching them avoids a round trip per chunk read.
_METADATA_SUFFIXES = (".zarray", ".zattrs", ".zgroup", ".zmetadata")


class FetchError(RuntimeError):
    """Raised when a store request fails for a reason other than 'not found'."""


def _urllib_fetch(url):
    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310 - same origin
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 403, 410):
            return None
        raise FetchError(f"{exc.code} for {url}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network dependent
        raise FetchError(f"{exc.reason} for {url}") from exc


def _xhr_fetch(url):  # pragma: no cover - requires a browser worker
    import js

    request = js.XMLHttpRequest.new()
    # Synchronous: only legal off the main thread, which is exactly where the
    # Python runtime lives.
    request.open("GET", url, False)
    request.responseType = "arraybuffer"
    request.send(None)

    if request.status in (404, 403, 410):
        # Genuinely absent: zarr reads an uninitialised chunk as its fill
        # value, which is the correct behaviour for a sparse array.
        return None
    if request.status >= 400 or request.status == 0:
        # Anything else is a broken request path, and must not be mistaken for
        # an empty chunk.
        raise FetchError(
            f"{request.status or 'network error'} for {url}: "
            f"{request.responseText[:200]}"
        )

    response = request.response
    if response is None:
        return b""
    return js.Uint8Array.new(response).to_bytes()


def _urllib_write(url, data):
    """Write (or, with ``data=None``, delete) through an HTTP request."""
    request = urllib.request.Request(
        url,
        data=b"" if data is None else bytes(data),
        method="DELETE" if data is None else "PUT",
    )
    try:
        with urllib.request.urlopen(request) as response:  # noqa: S310
            response.read()
    except urllib.error.HTTPError as exc:
        if data is None and exc.code in (404, 410):
            return  # deleting something absent is not a failure
        raise FetchError(f"{exc.code} for {url}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network dependent
        raise FetchError(f"{exc.reason} for {url}") from exc


def _xhr_write(url, data):  # pragma: no cover - requires a browser worker
    import js
    from pyodide.ffi import to_js

    request = js.XMLHttpRequest.new()
    request.open("DELETE" if data is None else "PUT", url, False)

    if data is None:
        request.send(None)
    else:
        request.setRequestHeader("Content-Type", "application/octet-stream")
        # `to_js` hands the buffer over without a further copy.
        request.send(to_js(bytes(data)))

    if data is None and request.status in (404, 410):
        return
    if request.status >= 400 or request.status == 0:
        raise FetchError(
            f"{request.status or 'network error'} writing {url}: "
            f"{request.responseText[:200]}"
        )


def default_fetch():
    """Return the read backend appropriate for the current runtime."""
    if is_pyodide():
        return _xhr_fetch
    return _urllib_fetch


def default_write():
    """Return the write backend appropriate for the current runtime."""
    if is_pyodide():
        return _xhr_write
    return _urllib_write


class HttpStoreBase:
    """Shared URL joining, caching and fetching for both zarr versions."""

    def __init__(self, base_url, fetch=None, write=None, writable=False):
        self.base_url = str(base_url).rstrip("/")
        self._fetch = fetch or default_fetch()
        self._write = write or (default_write() if writable else None)
        self._writable = writable or write is not None
        self._metadata_cache = {}

    def clear_cache(self):
        """Drop cached metadata documents (used on cache invalidation)."""
        self._metadata_cache.clear()

    def url_for(self, key):
        return f"{self.base_url}/{str(key).lstrip('/')}"

    def fetch_key(self, key):
        """Return the bytes stored under ``key``, or None when absent."""
        key = str(key).lstrip("/")
        cacheable = key.endswith(_METADATA_SUFFIXES)

        if cacheable and key in self._metadata_cache:
            return self._metadata_cache[key]

        data = self._fetch(self.url_for(key))

        if cacheable:
            self._metadata_cache[key] = data

        return data

    def write_key(self, key, value):
        """Store ``value`` under ``key``; ``None`` deletes it."""
        if not self._writable:
            raise NotImplementedError(
                f"{self.base_url} was opened read-only."
            )

        key = str(key).lstrip("/")
        self._write(self.url_for(key), value)

        # Metadata is cached for reads; keep that cache honest.
        if key.endswith(_METADATA_SUFFIXES):
            self._metadata_cache.pop(key, None)


class HttpZarrStoreV2(HttpStoreBase, MutableMapping):
    """zarr v2 (``MutableMapping``) store over an HTTP prefix."""

    def __getitem__(self, key):
        data = self.fetch_key(key)
        if data is None:
            raise KeyError(key)
        return data

    def __setitem__(self, key, value):
        self.write_key(key, value)

    def __delitem__(self, key):
        self.write_key(key, None)

    def __iter__(self):
        # Listing is not available over plain HTTP; zarr only needs it for
        # discovery, which OME-Zarr metadata makes unnecessary.
        return iter(())

    def __len__(self):
        return 0

    def __contains__(self, key):
        return self.fetch_key(key) is not None


def _make_v3_store_class():
    """Build the zarr v3 store class lazily (v3-only imports live inside)."""
    from zarr.abc.store import Store

    class HttpZarrStoreV3(HttpStoreBase, Store):
        def __init__(self, base_url, fetch=None, write=None, writable=False):
            HttpStoreBase.__init__(
                self, base_url, fetch=fetch, write=write, writable=writable
            )
            Store.__init__(self, read_only=not self._writable)

        def __eq__(self, other):
            return (
                isinstance(other, HttpZarrStoreV3)
                and other.base_url == self.base_url
            )

        def __hash__(self):
            return hash(("HttpZarrStoreV3", self.base_url))

        async def get(self, key, prototype, byte_range=None):
            data = self.fetch_key(key)
            if data is None:
                return None
            buffer = prototype.buffer.from_bytes(data)
            if byte_range is None:
                return buffer
            start = getattr(byte_range, "start", None) or 0
            end = getattr(byte_range, "end", None)
            return buffer[start:end] if end is not None else buffer[start:]

        async def get_partial_values(self, prototype, key_ranges):
            return [
                await self.get(key, prototype, byte_range)
                for key, byte_range in key_ranges
            ]

        async def exists(self, key):
            return self.fetch_key(key) is not None

        async def set(self, key, value):
            self.write_key(key, value.to_bytes())

        async def delete(self, key):
            self.write_key(key, None)

        async def list(self):
            return
            yield  # pragma: no cover - makes this an async generator

        async def list_dir(self, prefix):
            return
            yield  # pragma: no cover

        async def list_prefix(self, prefix):
            return
            yield  # pragma: no cover

        @property
        def supports_writes(self):
            return self._writable

        @property
        def supports_deletes(self):
            return self._writable

        @property
        def supports_listing(self):
            return False

        @property
        def supports_partial_writes(self):
            return False

    return HttpZarrStoreV3


_v3_store_class = None


def open_http_store(base_url, fetch=None, write=None, writable=False):
    """Return a zarr store rooted at ``base_url``.

    Read-only unless ``writable`` is set or a write backend is supplied.
    """
    global _v3_store_class

    if ZARR_V3:
        if _v3_store_class is None:
            _v3_store_class = _make_v3_store_class()
        return _v3_store_class(
            base_url, fetch=fetch, write=write, writable=writable
        )

    return HttpZarrStoreV2(
        base_url, fetch=fetch, write=write, writable=writable
    )


#: Path segment owned by the multiview-stitcher service worker. Any URL
#: containing it is served over HTTP; every other absolute path is an ordinary
#: filesystem path. Matching a segment rather than a prefix keeps this working
#: when the app is published under a sub-path (e.g. GitHub Pages project
#: sites), where a service worker may only claim URLs below its own directory.
SERVICE_WORKER_SEGMENT = "/__mvs__/"


def is_http_url(source, fetch=None):
    """True when ``source`` should be read over HTTP rather than as a path.

    The service-worker segment is the only thing that makes a root-relative
    path a URL. Treating *any* absolute path as served whenever a fetch
    backend happens to be configured would silently reroute ordinary
    filesystem inputs through it - the two are indistinguishable otherwise.
    """
    if not isinstance(source, str):
        return False
    if source.startswith(("http://", "https://")):
        return True
    return SERVICE_WORKER_SEGMENT in source


def resolve_zarr_source(url, fetch=None):
    """Return something ``zarr.open_group(store=...)`` accepts for ``url``.

    Local paths are passed through untouched so that the same code reads from
    an ordinary filesystem on CPython and from the service worker in the
    browser.
    """
    if is_http_url(url, fetch=fetch):
        return open_http_store(url, fetch=fetch)
    return url


def directory_fetch(root):
    """A fetch backend serving a local directory - the CPython test double.

    Mirrors what the service worker does in the browser: map a URL path onto a
    file below a granted directory and return its bytes, or None when missing.
    """
    import os

    root = str(root)

    def fetch(url):
        path = url.split("?", 1)[0]
        # Strip any scheme/host so both absolute and same-origin URLs work.
        if "://" in path:
            path = path.split("://", 1)[1]
            path = path[path.index("/") :] if "/" in path else "/"
        relative = path.lstrip("/")
        full = os.path.join(root, relative)
        if not os.path.isfile(full):
            return None
        with open(full, "rb") as handle:
            return handle.read()

    return fetch


def directory_write(root):
    """A write backend serving a local directory - the CPython test double.

    Mirrors what the fs worker does in the browser: map a URL path onto a file
    below a granted directory, write it whole, and close it.
    """
    import os
    import shutil

    root = str(root)

    def write(url, data):
        path = url.split("?", 1)[0]
        if "://" in path:
            path = path.split("://", 1)[1]
            path = path[path.index("/") :] if "/" in path else "/"
        full = os.path.join(root, path.lstrip("/"))

        if data is None:
            if os.path.isdir(full):
                shutil.rmtree(full)
            elif os.path.isfile(full):
                os.remove(full)
            return

        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as handle:
            handle.write(bytes(data))

    return write


def open_group_from_url(url, fetch=None):
    """Open an OME-Zarr root group from a URL or path."""
    source = resolve_zarr_source(url, fetch=fetch)
    if isinstance(source, str):
        return zarr.open_group(source, mode="r")
    return zarr.open_group(store=source, mode="r")
