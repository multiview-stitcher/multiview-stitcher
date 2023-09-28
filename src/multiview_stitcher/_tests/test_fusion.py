import warnings

import dask.array as da
import numpy as np
import xarray as xr

from multiview_stitcher import fusion, io, sample_data, spatial_image_utils
from multiview_stitcher.io import METADATA_TRANSFORM_KEY


def test_fuse_field():
    sims = io.read_mosaic_image_into_list_of_spatial_xarrays(
        sample_data.get_mosaic_sample_data_path()
    )

    for isim, sim in enumerate(sims):
        sims[isim] = spatial_image_utils.sim_sel_coords(
            sim, {"c": sim.coords["c"][0], "t": sim.coords["t"][0]}
        )

    params = [
        spatial_image_utils.get_affine_from_sim(
            sim, transform_key=METADATA_TRANSFORM_KEY
        )
        for sim in sims
    ]

    xfused = fusion.fuse_field(
        sims,
        params,
        output_origin=np.min(
            [
                spatial_image_utils.get_origin_from_sim(sim, asarray=True)
                for sim in sims
            ],
            0,
        ),
        output_spacing=spatial_image_utils.get_spacing_from_sim(
            sims[0], asarray=True
        ),
        output_shape=spatial_image_utils.get_shape_from_sim(
            xr.merge(sims), asarray=True
        ),
    )

    # check output is dask array and hasn't been converted into numpy array
    assert type(xfused.data) == da.core.Array
    assert xfused.dtype == sims[0].dtype


def test_fuse_sims():
    sims = io.read_mosaic_image_into_list_of_spatial_xarrays(
        sample_data.get_mosaic_sample_data_path()
    )

    # suppress pandas future warning occuring within xarray.concat
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)

        # test with two channels
        for isim, sim in enumerate(sims):
            sims[isim] = xr.concat([sim] * 2, dim="c").assign_coords(
                c=[sim.coords["c"].data[0], sim.coords["c"].data[0] + "_2"]
            )

    xfused = fusion.fuse(
        sims,
        transform_key=METADATA_TRANSFORM_KEY,
    )

    # check output is dask array and hasn't been converted into numpy array
    assert type(xfused.data) == da.core.Array
    assert xfused.dtype == sims[0].dtype

    # xfused.compute()
    xfused = xfused.compute(scheduled="threads")

    assert xfused.dtype == sims[0].dtype
    assert (
        METADATA_TRANSFORM_KEY
        in spatial_image_utils.get_tranform_keys_from_sim(xfused)
    )
