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
#: Counted in files, not scenes: a session holding every scene of one file
#: must not evict one scene's images to open the next.
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
def _file(path):
    """What is known of one CZI file: its kind, and the scenes read so far.

    Scenes are read on demand into ``scenes``, keyed by scene index.
    """
    return {"multiview": czi_utils.is_multiview_czi(path), "scenes": {}}


def _images(path, scene_index):
    """Every image of one CZI scene as spatial images (lazy, so this is cheap).

    Returns ``(sims, is_multiview)``. A multi-view acquisition is read as a
    whole, whatever the scene index.
    """
    entry = _file(path)
    multiview = entry["multiview"]
    key = 0 if multiview else scene_index

    if key not in entry["scenes"]:
        if multiview:
            sims = czi_utils.read_multiview_czi_into_sims(path)
            sims = tuple(_use_default_transform_key(sim) for sim in sims)
        else:
            sims = tuple(
                mvs_io.read_mosaic_into_sims_czifile(
                    path, scene_index=scene_index
                )
            )
        entry["scenes"][key] = sims

    return entry["scenes"][key], multiview


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
    _file.cache_clear()
    czi_utils.close_czi_files()


def _scene_label(scene):
    return scene["name"] or f"scene {scene['index']}"


def czi_scenes(path):
    """List the scenes of a CZI file, for the page to offer a choice.

    Returns ``{"multiview", "scenes"}``, ``scenes`` holding one
    ``{"index", "name", "n_images"}`` per scene. A multi-view acquisition is
    reported as a single scene, as it is read as a whole.
    """
    path = str(path)

    if _file(path)["multiview"]:
        sims, _ = _images(path, 0)
        return {
            "multiview": True,
            "scenes": [{"index": 0, "name": None, "n_images": len(sims)}],
        }

    return {
        "multiview": False,
        "scenes": [
            {
                "index": scene["index"],
                "name": scene["name"],
                "n_images": scene["n_tiles"],
            }
            for scene in czi_utils.get_czi_scenes(path)
        ],
    }


def czi_sources(path, scene_indices=(0,), name=None):
    """Describe every image of the given scenes of a CZI file as sources.

    Returns one ``{"url", "name"}`` per tile of a mosaic, or per view of a
    multi-view acquisition, scene by scene in the order given and within a
    scene in the order the file's reader returns them. When the file holds
    more than one scene, names carry the scene's, so that tiles of different
    scenes - possibly loaded at different times - can be told apart.
    """
    path = str(path)
    label = name or path.rstrip("/").split("/")[-1]
    described = czi_scenes(path)
    scenes = {scene["index"]: scene for scene in described["scenes"]}
    kind = "view" if described["multiview"] else "tile"

    if described["multiview"]:
        scene_indices = (0,)

    unknown = [i for i in scene_indices if int(i) not in scenes]
    if unknown:
        raise ValueError(
            f"'{label}' has no scene {unknown[0]}; its scenes are "
            f"{sorted(scenes)}."
        )

    sources = []
    for scene_index in scene_indices:
        scene_index = int(scene_index)
        sims, _ = _images(path, scene_index)
        prefix = (
            f"{label} {_scene_label(scenes[scene_index])}"
            if len(scenes) > 1
            else label
        )
        sources.extend(
            {
                "url": czi_url(path, index, scene_index),
                "name": f"{prefix} {kind} {index}",
            }
            for index in range(len(sims))
        )

    return sources


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
