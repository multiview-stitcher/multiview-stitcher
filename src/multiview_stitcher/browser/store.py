"""
Read-only zarr stores backed by HTTP.

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


def default_fetch():
    """Return the fetch backend appropriate for the current runtime."""
    if is_pyodide():
        return _xhr_fetch
    return _urllib_fetch


class HttpStoreBase:
    """Shared URL joining, caching and fetching for both zarr versions."""

    def __init__(self, base_url, fetch=None):
        self.base_url = str(base_url).rstrip("/")
        self._fetch = fetch or default_fetch()
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


class HttpZarrStoreV2(HttpStoreBase, MutableMapping):
    """zarr v2 (``MutableMapping``) store over an HTTP prefix."""

    def __getitem__(self, key):
        data = self.fetch_key(key)
        if data is None:
            raise KeyError(key)
        return data

    def __setitem__(self, key, value):
        raise NotImplementedError("HTTP-backed zarr store is read-only")

    def __delitem__(self, key):
        raise NotImplementedError("HTTP-backed zarr store is read-only")

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
        def __init__(self, base_url, fetch=None):
            Store.__init__(self, read_only=True)
            HttpStoreBase.__init__(self, base_url, fetch=fetch)

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
            raise NotImplementedError("HTTP-backed zarr store is read-only")

        async def delete(self, key):
            raise NotImplementedError("HTTP-backed zarr store is read-only")

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
            return False

        @property
        def supports_deletes(self):
            return False

        @property
        def supports_listing(self):
            return False

        @property
        def supports_partial_writes(self):
            return False

    return HttpZarrStoreV3


_v3_store_class = None


def open_http_store(base_url, fetch=None):
    """Return a read-only zarr store rooted at ``base_url``."""
    global _v3_store_class

    if ZARR_V3:
        if _v3_store_class is None:
            _v3_store_class = _make_v3_store_class()
        return _v3_store_class(base_url, fetch=fetch)

    return HttpZarrStoreV2(base_url, fetch=fetch)


#: Path segment owned by the multiview-stitcher service worker. Any URL
#: containing it is served over HTTP; every other absolute path is an ordinary
#: filesystem path. Matching a segment rather than a prefix keeps this working
#: when the app is published under a sub-path (e.g. GitHub Pages project
#: sites), where a service worker may only claim URLs below its own directory.
SERVICE_WORKER_SEGMENT = "/__mvs__/"


def is_http_url(source, fetch=None):
    """True when ``source`` should be read over HTTP rather than as a path."""
    if not isinstance(source, str):
        return False
    if source.startswith(("http://", "https://")):
        return True
    if SERVICE_WORKER_SEGMENT in source:
        return True
    # An explicit fetch backend means the caller has already decided that this
    # source is served rather than opened.
    return fetch is not None and source.startswith("/")


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


def open_group_from_url(url, fetch=None):
    """Open an OME-Zarr root group from a URL or path."""
    source = resolve_zarr_source(url, fetch=fetch)
    if isinstance(source, str):
        return zarr.open_group(source, mode="r")
    return zarr.open_group(store=source, mode="r")
