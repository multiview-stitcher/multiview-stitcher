import os
import tempfile
from pathlib import Path

import multiview_stitcher.spatial_image_utils as si_utils
from multiview_stitcher import fusion, msi_utils, param_utils, sample_data

# to be used in the future when we have test data to be cached
# @pytest.fixture(scope="session")
# def testdata_dir_path() -> Path:
#     TEST_DIR = Path(__file__).parent
#     if not TEST_DIR.exists():
#         raise FileNotFoundError(f"Test data directory not found: {TEST_DIR}")
#     return TEST_DIR / "data/"


def create_input(
    test_params=None,
    tmp_dir="/tmp/test_benchmarking",
):
    """
    Test that the circumvent_dask_for_zarr_backed_input flag works as expected.
    """

    # create minimal example grid with overlap
    if test_params is None:
        test_params = {"t": 1, "c": 1, "ndim": 3, "input_chunking": 10}
    ndim = test_params["ndim"]
    sims = sample_data.generate_tiled_dataset(
        ndim=ndim,
        overlap=50,
        N_c=test_params["c"],
        N_t=test_params["t"],
        tile_size=300,
        tiles_x=3,
        tiles_y=3,
        tiles_z=1,
        spacing_x=1,
        spacing_y=1,
        spacing_z=1,
    )

    # rechunk input data
    for sim in sims:
        sim.data = sim.data.rechunk(test_params["input_chunking"])
        p = param_utils.affine_to_xaffine(
            param_utils.random_affine(
                ndim=ndim,
                rotation_scale=0,
                scale_scale=0,
                translation_scale=10,
            )
        )
        si_utils.set_sim_affine(sim, p, transform_key="random")

    # persist input data to zarr
    for iview in range(len(sims)):
        msi_utils.multiscale_spatial_image_to_zarr(
            msi_utils.get_msim_from_sim(sims[iview], scale_factors=[]),
            os.path.join(tmp_dir, f"msim_view_{iview}.zarr"),
        )
    return


def fuse_zarr_to_zarr(
    tmp_dir="/tmp/test_benchmarking",
):
    msims = [
        msi_utils.multiscale_spatial_image_from_zarr(
            os.path.join(tmp_dir, f"msim_view_{iview}.zarr")
        )
        for iview in range(2)
    ]

    sims = [msi_utils.get_sim_from_msim(msim) for msim in msims]

    sdims = si_utils.get_spatial_dims_from_sim(sims[0])

    fused_sim = fusion.fuse(
        sims,
        # transform_key=METADATA_TRANSFORM_KEY,
        transform_key="random",
        output_chunksize={dim: 8 for dim in sdims},
    )

    fused_sim.data.to_zarr(
        os.path.join(tmp_dir, "fused_sim.zarr"),
        overwrite=True,
        return_stored=True,
        compute=True,
    )

    return fused_sim


def test_fusion(
    benchmark,
    # testdata_dir_path=None,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        testdata_dir_path = Path(tmpdir)

        create_input(
            tmp_dir=testdata_dir_path,
        )

        # benchmark something
        benchmark.pedantic(
            fuse_zarr_to_zarr,
            args=(testdata_dir_path,),
            # setup=create_input,
            iterations=1,
            rounds=3,
        )
