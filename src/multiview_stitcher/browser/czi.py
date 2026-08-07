"""Opening CZI files as browser sources.

A CZI holds a whole dataset in one file, while the rest of the browser
addresses one view per source URL. This module bridges the two: it enumerates a
file's images as URLs, and opens any one of them on demand.

Two kinds of CZI are read, by the readers the library already has:

* a **mosaic**, whose tiles are laid out in a plane, through
  :func:`multiview_stitcher.io.read_mosaic_into_sims_czifile`;
* a **multi-view** acquisition, whose views are stacks recorded at different
  angles, through
  :func:`multiview_stitcher.czi_utils.read_multiview_czi_into_sims`.

Both are used unchanged - the same code paths CPython uses. What makes them
work in Pyodide is where the bytes come from: the page mounts the user's
``File`` through Emscripten's WORKERFS, which serves reads lazily from the file
on disk by way of ``Blob.slice``, so an ordinary path reaches an ordinary
seekable file and the whole multi-gigabyte CZI is never held in memory. See
``mountFiles`` in docs/browser/py-runtime.js.

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

from multiview_stitcher import czi_utils, msi_utils
from multiview_stitcher import io as mvs_io
from multiview_stitcher import spatial_image_utils as si_utils

#: URL scheme identifying one image of a CZI file, e.g.
#: ``mvs-czi:/czi/a1b2c3/mosaic.czi?scene=0&index=2``. ``index`` counts tiles of
#: a mosaic or views of a multi-view acquisition, in the order its reader
#: returns them; ``scene`` applies to mosaics only. The path is a path in the
#: Python runtime's own filesystem, which in the browser is where the page
#: mounted the file.
SCHEME = "mvs-czi:"

#: How many CZI files' image lists a worker keeps. Opening one image reads the
#: whole file's metadata, so they are built together and reused; a compute
#: worker rebuilding a session opens every image of the same file in a row.
_CACHE_SIZE = 2

#: Coordinate system `read_multiview_czi_into_sims` writes the view positions
#: into. Everything else in the browser - OME-Zarr inputs, mosaic CZI, the
#: generated examples - uses `si_utils.DEFAULT_TRANSFORM_KEY`, and a session
#: only offers the transform keys *all* of its views share, so a view arriving
#: under a name of its own would leave the session with none in common.
_MULTIVIEW_TRANSFORM_KEY = "metadata"


def is_czi_url(url):
    """Is ``url`` a reference to one image of a CZI file?"""
    return isinstance(url, str) and url.startswith(SCHEME)


def czi_url(path, index, scene_index=0):
    """Build the URL addressing one image of a CZI file."""
    return f"{SCHEME}{path}?scene={int(scene_index)}&index={int(index)}"


def parse_czi_url(url):
    """Split a CZI image URL into ``(path, scene_index, index)``."""
    if not is_czi_url(url):
        raise ValueError(f"'{url}' is not a CZI image URL.")

    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    if not parsed.path:
        raise ValueError(f"'{url}' names no CZI file.")

    return (
        parsed.path,
        int(query.get("scene", ["0"])[0]),
        int(query.get("index", ["0"])[0]),
    )


@lru_cache(maxsize=_CACHE_SIZE)
def _images(path, scene_index):
    """Every image of one CZI as spatial images (lazy, so this is cheap).

    Returns ``(sims, is_multiview)``.
    """
    if czi_utils.is_multiview_czi(path):
        sims = czi_utils.read_multiview_czi_into_sims(path)
        return tuple(_use_default_transform_key(sim) for sim in sims), True

    sims = mvs_io.read_mosaic_into_sims_czifile(path, scene_index=scene_index)
    return tuple(sims), False


def _use_default_transform_key(sim):
    """Rename the multi-view reader's coordinate system to the standard one.

    Renamed here rather than in the reader because ``metadata`` is the name its
    callers - notebooks, scripts - already pass to `plot_positions`, `register`
    and the rest.
    """
    transforms = sim.attrs.get("transforms", {})

    if (
        _MULTIVIEW_TRANSFORM_KEY in transforms
        and si_utils.DEFAULT_TRANSFORM_KEY not in transforms
    ):
        transforms[si_utils.DEFAULT_TRANSFORM_KEY] = transforms.pop(
            _MULTIVIEW_TRANSFORM_KEY
        )

    return sim


def forget_files():
    """Drop cached image lists and open file handles.

    Called when a session is cleared: a mount the page has released must not be
    kept alive by a cached handle.
    """
    _images.cache_clear()
    czi_utils.close_czi_files()


def czi_sources(path, scene_index=0, name=None):
    """Describe every image of a CZI file as a loadable source.

    Returns one ``{"url", "name"}`` per tile of a mosaic, or per view of a
    multi-view acquisition, in the order the file's reader returns them.
    """
    sims, multiview = _images(str(path), int(scene_index))
    label = name or str(path).rstrip("/").split("/")[-1]
    kind = "view" if multiview else "tile"

    return [
        {
            "url": czi_url(path, index, scene_index),
            "name": f"{label} {kind} {index}",
        }
        for index in range(len(sims))
    ]


def build_sim(url):
    """Open the one image a CZI URL addresses."""
    path, scene_index, index = parse_czi_url(url)
    sims, _ = _images(path, scene_index)

    if not 0 <= index < len(sims):
        raise ValueError(
            f"'{path}' holds {len(sims)} image(s) in scene {scene_index}; "
            f"asked for index {index}."
        )

    # Copied so that a caller setting transforms on the returned image cannot
    # write through to the cached one shared with every other view.
    return sims[index].copy(deep=False)


def build_msim(url, scale_factors=None):
    """Open one image as a multiscale image, matching an OME-Zarr input.

    Mosaic tiles and multi-view stacks alike get the pyramid msi_utils derives
    from the image's shape, and the chunking stays the CZI's own - one chunk
    per subblock, the smallest unit the file can be read in.
    """
    return msi_utils.get_msim_from_sim(
        build_sim(url), scale_factors=scale_factors
    )
