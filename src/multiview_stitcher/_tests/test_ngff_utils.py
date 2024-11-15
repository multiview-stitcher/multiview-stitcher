import tempfile

import ngff_zarr
import pytest

from multiview_stitcher import io, msi_utils, ngff_utils, sample_data


@pytest.mark.parametrize(
    "ndim",
    [2, 3],
)
def test_round_trip(ndim):
    sim = sample_data.generate_tiled_dataset(
        ndim=ndim,
        overlap=0,
        N_c=1,
        N_t=1,
        tile_size=30,
        tiles_x=1,
        tiles_y=1,
        tiles_z=1,
        spacing_x=1,
        spacing_y=1,
        spacing_z=1,
    )[0]

    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[2, 4])

    with tempfile.TemporaryDirectory() as zarr_path:
        ngff_multiscales = ngff_utils.msim_to_ngff_multiscales(
            msim, transform_key=io.METADATA_TRANSFORM_KEY
        )
        ngff_zarr.to_ngff_zarr(zarr_path, ngff_multiscales)
        ngff_utils.ngff_multiscales_to_msim(
            ngff_zarr.from_ngff_zarr(zarr_path),
            transform_key=io.METADATA_TRANSFORM_KEY,
        )

    import pdb

    pdb.set_trace()
    # assert msim_read.equals(msim)
