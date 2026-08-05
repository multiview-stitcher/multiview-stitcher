"""
Sample-data datasets for trying the browser app without any local files.

Generation is deterministic: every worker that is handed the same source URL
reproduces byte-identical tiles from the seed encoded in it. That matters
because compute workers rebuild their own copy of a session rather than
receiving image data, so a generator seeded from global RNG state would make
them silently disagree about the pixels they register and fuse.

The tiles come from :func:`sample_data.generate_tiled_dataset`, the same
generator used by the library's examples and tests.
"""

from functools import cache

import dask.array as da

from multiview_stitcher import msi_utils, sample_data
from multiview_stitcher import spatial_image_utils as si_utils

#: URL scheme identifying a generated dataset, e.g. ``mvs-example:tiles-3d/2``.
SCHEME = "mvs-example:"


#: Examples offered by the browser, in menu order.
EXAMPLE_MENU = (
    "tiles-3d-1c",
    "tiles-3d-2c",
    "tiles-2d-1c",
    "tiles-2d-2c",
    "tiles-2d-3t-2c",
)


#: Available examples, keyed by the name that appears in the URL. ``tiles-3d``
#: remains as a non-menu alias for old links and tests.
EXAMPLES = {
    "tiles-3d-1c": {
        "label": "3D · single channel · 2×2",
        "ndim": 3,
        "n_channels": 1,
        "tile_size": 64,
        "overlap": 16,
        "shift_scale": 8.0,
        "seed": 0,
    },
    "tiles-3d-2c": {
        "label": "3D · two channels · 2×2",
        "ndim": 3,
        "n_channels": 2,
        "tile_size": 64,
        "overlap": 16,
        "shift_scale": 8.0,
        "seed": 1,
    },
    "tiles-2d-1c": {
        "label": "2D · single channel · 2×2",
        "ndim": 2,
        "n_channels": 1,
        "tile_size": 128,
        "overlap": 32,
        "shift_scale": 8.0,
        "seed": 2,
    },
    "tiles-2d-2c": {
        "label": "2D · two channels · 2×2",
        "ndim": 2,
        "n_channels": 2,
        "tile_size": 128,
        "overlap": 32,
        "shift_scale": 8.0,
        "seed": 3,
    },
    # The only example with a time axis: manual placement can be restricted to
    # a range of timepoints, and a transform that varies over t has to survive
    # registration, fusion and the round trip through the viewer.
    "tiles-2d-3t-2c": {
        "label": "2D · 3 timepoints · two channels · 2×2",
        "ndim": 2,
        "n_channels": 2,
        "n_timepoints": 3,
        "tile_size": 128,
        "overlap": 32,
        "shift_scale": 8.0,
        "seed": 4,
    },
}

# Keep the non-menu legacy fixture compact: many Python tests use this alias
# to exercise the browser pipeline, while the four examples offered in the UI
# use the larger, more visibly misaligned data above.
EXAMPLES["tiles-3d"] = {
    **EXAMPLES["tiles-3d-1c"],
    "label": "3D · 2×2 tiles",
    "tile_size": 32,
    "overlap": 8,
    "shift_scale": 3.0,
}


def is_example_url(url):
    return isinstance(url, str) and url.startswith(SCHEME)


def parse_example_url(url):
    """Split ``mvs-example:<name>/<tile index>`` into its parts."""
    if not is_example_url(url):
        raise ValueError(f"'{url}' is not an example dataset URL.")

    body = url[len(SCHEME) :]
    name, _, index = body.partition("/")

    if name not in EXAMPLES:
        raise ValueError(
            f"Unknown example '{name}'. Available: {sorted(EXAMPLES)}."
        )

    return name, int(index or 0)


def example_sources(name):
    """The source URLs and display names of one example dataset."""
    if name not in EXAMPLES:
        raise ValueError(f"Unknown example '{name}'.")
    n_tiles = 4

    return [
        {"url": f"{SCHEME}{name}/{index}", "name": f"{name} tile {index}"}
        for index in range(n_tiles)
    ]


@cache
def _dataset(name):
    """Generate and cache one deterministic 2×2 sample-data dataset."""
    spec = EXAMPLES[name]
    # ``sample_data`` uses Dask's random generator for channel ground truth.
    # Seed it so independent browser workers reconstruct identical URL data.
    da.random.seed(spec["seed"])
    return tuple(
        sample_data.generate_tiled_dataset(
            ndim=spec["ndim"],
            N_c=spec["n_channels"],
            N_t=spec.get("n_timepoints", 1),
            tile_size=spec["tile_size"],
            tiles_x=2,
            tiles_y=2,
            tiles_z=1,
            overlap=spec["overlap"],
            zoom=6,
            chunksize=spec["tile_size"],
            spacing_x=0.5,
            spacing_y=0.5,
            spacing_z=2.0,
            shift_scale=spec["shift_scale"],
            drift_scale=0.0,
            transform_key=si_utils.DEFAULT_TRANSFORM_KEY,
        )
    )


def build_sim(name, tile_index):
    """Build one tile of an example dataset as a spatial image."""
    sims = _dataset(name)
    if not 0 <= tile_index < len(sims):
        raise ValueError(
            f"Example '{name}' has {len(sims)} tiles; asked for "
            f"{tile_index}."
        )
    return sims[tile_index].copy(deep=False)


def build_msim(name, tile_index, scale_factors=None):
    """Build one tile as a multiscale image, matching an OME-Zarr input."""
    sim = build_sim(name, tile_index)

    if scale_factors is None:
        sdims = si_utils.get_spatial_dims_from_sim(sim)
        # One extra level, halving only the in-plane dimensions - the same
        # shape of pyramid a small OME-Zarr tile would carry.
        scale_factors = [{dim: 2 for dim in sdims if dim != "z"}]

    return msi_utils.get_msim_from_sim(sim, scale_factors=scale_factors)
