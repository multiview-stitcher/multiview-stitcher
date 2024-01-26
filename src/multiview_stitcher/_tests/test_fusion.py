import warnings

import dask.array as da
import xarray as xr

from multiview_stitcher import fusion, io, sample_data, spatial_image_utils
from multiview_stitcher.io import METADATA_TRANSFORM_KEY


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
