import pytest

from multiview_stitcher import sample_data


@pytest.mark.parametrize(
    "ndim, overlap, N_t, N_c",
    [
        (ndim, overlap, N_t, N_c)
        for ndim in [2, 3]
        for overlap in [0, 2]
        for N_t in [1, 2]
        for N_c in [1, 2]
    ],
)
def test_generate_tiled_dataset(ndim, overlap, N_t, N_c):
    """
    Could be much more general
    """

    tile_size = 10
    spacing_x = 0.5
    spacing_y = 0.5
    spacing_z = 0.5
    sims = sample_data.generate_tiled_dataset(
        ndim=ndim,
        overlap=overlap,
        N_c=N_c,
        N_t=N_t,
        tile_size=tile_size,
        tiles_x=1,
        tiles_y=2,
        tiles_z=1,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        spacing_z=spacing_z,
        drift_scale=0,
        shift_scale=0,
    )

    assert sims[0].data.ndim == ndim + 2
