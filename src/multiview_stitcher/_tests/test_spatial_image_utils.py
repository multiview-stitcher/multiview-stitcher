import dask.array as da
import numpy as np
import pytest

from multiview_stitcher import param_utils
from multiview_stitcher import spatial_image_utils as si_utils


@pytest.mark.parametrize(
    "xp, ndim", [(xp, ndim) for xp in [np, da] for ndim in [2, 3]]
)
def test_sim_array_input_backends(xp, ndim):
    sim = si_utils.get_sim_from_array(
        xp.ones((5,) * ndim),
        dims=si_utils.SPATIAL_DIMS[-ndim:],
        scale={dim: 1.0 for dim in si_utils.SPATIAL_DIMS[-ndim:]},
        translation={dim: -1.0 for dim in si_utils.SPATIAL_DIMS[-ndim:]},
        affine=param_utils.identity_transform(ndim),
    )

    assert isinstance(sim.data, da.Array)


def test_max_project():
    ndim = 3
    dim = "z"
    transform_key = "test"

    sim = si_utils.get_sim_from_array(
        np.ones((5,) * ndim),
        dims=si_utils.SPATIAL_DIMS[-ndim:],
        scale={dim: 1.0 for dim in si_utils.SPATIAL_DIMS[-ndim:]},
        translation={dim: -1.0 for dim in si_utils.SPATIAL_DIMS[-ndim:]},
        affine=param_utils.identity_transform(ndim),
        transform_key=transform_key,
    )

    sim_proj = si_utils.max_project_sim(sim, dim)

    assert dim not in sim_proj.dims

    affine_sim = si_utils.get_affine_from_sim(sim, transform_key)

    for pdim in ["x_in", "x_out"]:
        assert dim in affine_sim.coords[pdim].values

    affine_sim_proj = si_utils.get_affine_from_sim(sim_proj, transform_key)

    for pdim in ["x_in", "x_out"]:
        assert dim not in affine_sim_proj.coords[pdim].values
