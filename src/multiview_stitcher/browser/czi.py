"""Opening mosaic CZI files as browser sources.

A CZI holds every tile of a mosaic in one file, while the rest of the browser
addresses one view per source URL. This module bridges the two: it enumerates a
file's tiles as URLs, and opens any one of them on demand.

The reader itself is :func:`multiview_stitcher.io.read_mosaic_into_sims_czifile`
unchanged - the same code path CPython uses. What makes it work in Pyodide is
where the bytes come from: the page mounts the user's ``File`` through
Emscripten's WORKERFS, which serves reads lazily from the file on disk by way of
``Blob.slice``, so an ordinary path reaches an ordinary seekable file and the
whole multi-gigabyte CZI is never held in memory. See ``mountCziFiles`` in
docs/browser/py-runtime.js.

Every worker opens the file for itself, exactly as it re-opens an OME-Zarr, so a
source is fully described by its URL and nothing but URLs crosses the worker
boundary.

Only uncompressed CZI files can be read here. Decoding the compressed variants
(LZW, JPEG, JPEG XR, ZSTD) needs imagecodecs, which is a C extension with no
WebAssembly build; :mod:`multiview_stitcher.czifile_patch` raises a message
saying so rather than failing obscurely.
"""

from functools import lru_cache
from urllib.parse import parse_qs, urlparse

from multiview_stitcher import io as mvs_io
from multiview_stitcher import msi_utils

#: URL scheme identifying one tile of a mosaic CZI, e.g.
#: ``mvs-czi:/czi/a1b2c3/mosaic.czi?scene=0&tile=2``. The path is a path in the
#: Python runtime's own filesystem, which in the browser is where the page
#: mounted the file.
SCHEME = "mvs-czi:"

#: How many CZI files' tile lists a worker keeps. Opening one tile reads the
#: whole file's metadata, so the tiles are built together and reused; a compute
#: worker rebuilding a session opens every tile of the same file in a row.
_CACHE_SIZE = 2


def is_czi_url(url):
    """Is ``url`` a reference to one tile of a CZI file?"""
    return isinstance(url, str) and url.startswith(SCHEME)


def czi_url(path, tile_index, scene_index=0):
    """Build the URL addressing one tile of a mosaic CZI."""
    return f"{SCHEME}{path}?scene={int(scene_index)}&tile={int(tile_index)}"


def parse_czi_url(url):
    """Split a CZI tile URL into ``(path, scene_index, tile_index)``."""
    if not is_czi_url(url):
        raise ValueError(f"'{url}' is not a CZI tile URL.")

    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    if not parsed.path:
        raise ValueError(f"'{url}' names no CZI file.")

    return (
        parsed.path,
        int(query.get("scene", ["0"])[0]),
        int(query.get("tile", ["0"])[0]),
    )


@lru_cache(maxsize=_CACHE_SIZE)
def _tiles(path, scene_index):
    """Every tile of one scene, as spatial images (lazy, so this is cheap)."""
    return tuple(
        mvs_io.read_mosaic_into_sims_czifile(path, scene_index=scene_index)
    )


def forget_files():
    """Drop cached tile lists and open file handles.

    Called when a session is cleared: a mount the page has released must not be
    kept alive by a cached handle.
    """
    from multiview_stitcher import czi_utils

    _tiles.cache_clear()
    czi_utils.close_czi_files()


def czi_sources(path, scene_index=0, name=None):
    """Describe every tile of a mosaic CZI as a loadable source.

    Returns one ``{"url", "name"}`` per tile, in the file's own tile order.
    """
    tiles = _tiles(str(path), int(scene_index))
    label = name or str(path).rstrip("/").split("/")[-1]

    return [
        {
            "url": czi_url(path, index, scene_index),
            "name": f"{label} tile {index}",
        }
        for index in range(len(tiles))
    ]


def build_sim(url):
    """Open the one tile a CZI tile URL addresses."""
    path, scene_index, tile_index = parse_czi_url(url)
    tiles = _tiles(path, scene_index)

    if not 0 <= tile_index < len(tiles):
        raise ValueError(
            f"'{path}' has {len(tiles)} tile(s) in scene {scene_index}; "
            f"asked for tile {tile_index}."
        )

    # Copied so that a caller setting transforms on the returned image cannot
    # write through to the cached one shared with every other view.
    return tiles[tile_index].copy(deep=False)


def build_msim(url, scale_factors=None):
    """Open one tile as a multiscale image, matching an OME-Zarr input.

    With no ``scale_factors`` the pyramid is the one msi_utils derives from the
    tile's shape, and the chunking stays the CZI's own: one chunk per subblock,
    which is the smallest unit the file can be read in.
    """
    return msi_utils.get_msim_from_sim(
        build_sim(url), scale_factors=scale_factors
    )
