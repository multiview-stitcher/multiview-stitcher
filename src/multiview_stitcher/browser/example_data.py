"""
Synthetic example datasets for trying the browser app without any data.

Generation is deterministic: every worker that is handed the same source URL
reproduces byte-identical tiles from the seed encoded in it. That matters
because compute workers rebuild their own copy of a session rather than
receiving image data, so a generator seeded from global RNG state would make
them silently disagree about the pixels they register and fuse.

The tiles are produced with a nominal (unshifted) origin while the pixels are
sampled at a shifted position, so registration has a known offset to recover.
"""

import numpy as np
from scipy import ndimage

from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils

#: URL scheme identifying a generated dataset, e.g. ``mvs-example:tiles-3d/2``.
SCHEME = "mvs-example:"


#: Available examples, keyed by the name that appears in the URL.
EXAMPLES = {
    "tiles-3d": {
        "label": "3D · 2×2 tiles",
        "ndim": 3,
        "grid": {"y": 2, "x": 2},
        "tile_shape": {"z": 16, "y": 64, "x": 64},
        "overlap": {"y": 16, "x": 16},
        "spacing": {"z": 2.0, "y": 0.5, "x": 0.5},
        "shift": 3,  # px, the offset registration should recover
        "seed": 0,
    },
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
    spec = EXAMPLES[name]
    n_tiles = int(np.prod(list(spec["grid"].values())))

    return [
        {"url": f"{SCHEME}{name}/{index}", "name": f"{name} tile {index}"}
        for index in range(n_tiles)
    ]


def _tile_positions(spec):
    """Nominal top-left corner of each tile, in pixels, in row-major order."""
    dims = list(spec["grid"])
    steps = {
        dim: spec["tile_shape"][dim] - spec["overlap"][dim] for dim in dims
    }

    positions = []
    for index in np.ndindex(*(spec["grid"][dim] for dim in dims)):
        positions.append(
            {dim: int(i * steps[dim]) for dim, i in zip(dims, index)}
        )
    return positions


def _ground_truth(spec):
    """A smooth, blobby volume large enough to cover the whole tile grid."""
    rng = np.random.default_rng(spec["seed"])
    positions = _tile_positions(spec)
    margin = spec["shift"] + 1

    shape = {
        dim: (
            max(position[dim] for position in positions)
            + spec["tile_shape"][dim]
            + 2 * margin
            if dim in spec["grid"]
            else spec["tile_shape"][dim]
        )
        for dim in spec["tile_shape"]
    }

    dims = list(spec["tile_shape"])
    volume = np.zeros([shape[dim] for dim in dims], dtype=np.float32)

    # Sparse bright points blurred into blobs: plenty of texture for phase
    # correlation, and cheap to generate.
    n_blobs = max(64, int(np.prod(volume.shape) // 400))
    coords = tuple(
        rng.integers(0, size, n_blobs) for size in volume.shape
    )
    volume[coords] = rng.uniform(300.0, 4000.0, n_blobs)

    sigma = [1.0 if dim == "z" else 2.0 for dim in dims]
    volume = ndimage.gaussian_filter(volume, sigma=sigma)
    volume = volume / max(volume.max(), 1e-6) * 4000.0

    return volume.astype(np.uint16), shape


def build_sim(name, tile_index):
    """Build one tile of an example dataset as a spatial image."""
    spec = EXAMPLES[name]
    volume, _ = _ground_truth(spec)

    dims = list(spec["tile_shape"])
    positions = _tile_positions(spec)
    if not 0 <= tile_index < len(positions):
        raise ValueError(
            f"Example '{name}' has {len(positions)} tiles; asked for "
            f"{tile_index}."
        )

    position = positions[tile_index]
    margin = spec["shift"] + 1

    # Each tile is sampled a few pixels away from where its metadata says it
    # is, which is the offset a registration has to find.
    rng = np.random.default_rng(spec["seed"] + 1000 + tile_index)
    shifts = {
        dim: int(rng.integers(-spec["shift"], spec["shift"] + 1))
        for dim in spec["grid"]
    }

    slices = []
    for dim in dims:
        start = margin + position.get(dim, 0) + shifts.get(dim, 0)
        slices.append(slice(start, start + spec["tile_shape"][dim]))

    data = np.ascontiguousarray(volume[tuple(slices)])

    return si_utils.get_sim_from_array(
        data,
        dims=dims,
        scale={dim: spec["spacing"][dim] for dim in dims},
        # The stored origin is the nominal grid position, i.e. it does not
        # include the shift applied to the pixels above.
        translation={
            dim: position.get(dim, 0) * spec["spacing"][dim] for dim in dims
        },
        transform_key=si_utils.DEFAULT_TRANSFORM_KEY,
    )


def build_msim(name, tile_index, scale_factors=None):
    """Build one tile as a multiscale image, matching an OME-Zarr input."""
    sim = build_sim(name, tile_index)

    if scale_factors is None:
        sdims = si_utils.get_spatial_dims_from_sim(sim)
        # One extra level, halving only the in-plane dimensions - the same
        # shape of pyramid a small OME-Zarr tile would carry.
        scale_factors = [{dim: 2 for dim in sdims if dim != "z"}]

    return msi_utils.get_msim_from_sim(sim, scale_factors=scale_factors)
