from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
from scipy import ndimage

from multiview_stitcher import param_utils
from multiview_stitcher.io import METADATA_TRANSFORM_KEY


def get_mosaic_sample_data_path():
    sample_path = (
        Path(__file__).parent.parent.parent
        / "test-datasets"
        / "mosaic_test.czi"
    )

    return sample_path


def generate_tiled_dataset(
    ndim=2,
    N_c=2,
    N_t=20,
    tile_size=30,
    tiles_x=2,
    tiles_y=2,
    tiles_z=1,
    overlap=5,
    zoom=6,
    random_data=False,
    chunksize=128,
    dtype=np.uint16,
    spacing_x=0.5,
    spacing_y=0.5,
    spacing_z=2.0,
    shift_scale=2.0,
    drift_scale=2.0,
    transform_key=METADATA_TRANSFORM_KEY,
):
    def transform_input(
        x, shifts, drifts, im_gt, overlap=0, zoom=10.0, block_info=None
    ):
        x = x.squeeze()

        output_shape = np.array(x.shape)

        shift = shifts[block_info[0]["chunk-location"]]
        drift = drifts[block_info[0]["chunk-location"]]

        eff_shape = output_shape - overlap
        offset = np.array(block_info[None]["chunk-location"][1:]) * eff_shape

        offset = offset + drift + shift

        offset = offset / zoom

        x = ndimage.affine_transform(
            im_gt,
            matrix=np.eye(x.ndim) / zoom,
            offset=offset,
            output_shape=output_shape,
            mode="reflect",
        )[None]

        return x

    # build the array
    tiles = da.empty(
        (N_t,)
        + tuple([tile_size * f for f in [tiles_z, tiles_y, tiles_x][-ndim:]]),
        chunks=(1,) + (tile_size,) * ndim,
        dtype=dtype,
    )

    # simulate shifts and drifts
    shifts = (np.random.random(tiles.numblocks + (ndim,)) - 0.5) * shift_scale
    drifts = np.cumsum(
        np.ones(tiles.numblocks + (ndim,)) * drift_scale, axis=0
    )

    np.random.seed(0)
    tls = []
    for _ch in range(N_c):
        # the channel ground truth
        im_gt = da.random.randint(
            0,
            100,
            [
                2 * f * tile_size // zoom
                for f in [tiles_z, tiles_y, tiles_x][-ndim:]
            ],
            dtype=np.uint16,
        )

        if random_data:
            tl = da.random.randint(
                0, 200, tiles.shape, dtype=dtype, chunks=tiles.chunks
            )
        else:
            tl = tiles.map_blocks(
                transform_input,
                shifts,
                drifts,
                im_gt,
                zoom=zoom,
                overlap=overlap,
                dtype=tiles.dtype,
            )

        tls.append(tl[None])

    tls = da.concatenate(tls, axis=0)

    # generate sims
    sims = []
    spatial_dims = ["z", "y", "x"][-ndim:]
    spacing = [spacing_z, spacing_y, spacing_x][-ndim:]
    for tile_index in np.ndindex(tls.numblocks[2:]):
        tile_index = np.array(tile_index)
        tile = tls.blocks[
            tuple(
                [slice(0, N_c), slice(0, N_t)]
                + [slice(ti, ti + 1) for ti in tile_index]
            )
        ]
        origin = (
            tile_index * tile_size * spacing - overlap * (tile_index) * spacing
        )
        sim = xr.DataArray(
            tile,
            dims=["c", "t"] + spatial_dims,
            # For python >= 3.9 we can use the union '|' operator to merge to dict
            coords={
                **{
                    spatial_dims[dim]:
                    # origin[dim] +\
                    np.arange(tile.shape[2 + dim]) * spacing[dim]
                    for dim in range(ndim)
                },
                "c": ["channel " + str(c) for c in range(N_c)],
            },
        )

        affine = param_utils.affine_from_translation(origin)

        affine_xr = xr.DataArray(
            np.stack([affine] * len(sim.coords["t"])),
            dims=["t", "x_in", "x_out"],
        )

        sim.attrs["transforms"] = xr.Dataset({transform_key: affine_xr})

        # sim.name = 'tile_' + '_'.join([str(ti) for ti in tile_index])
        sim.data = sim.data.rechunk((1, 1) + (chunksize,) * ndim)

        sims.append(sim)

    return sims
