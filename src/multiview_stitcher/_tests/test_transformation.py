import dask.array as da
import numpy as np

from multiview_stitcher import param_utils, transformation
from multiview_stitcher import spatial_image_utils as si_utils


def test_transform_sim_skips_affine_for_noop(monkeypatch):
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    sim = si_utils.get_sim_from_array(
        da.from_array(data, chunks=(2, 2)),
        dims=["y", "x"],
        scale={"y": 0.5, "x": 2.0},
        translation={"y": 3.0, "x": -4.0},
    )
    output_stack_properties = si_utils.get_stack_properties_from_sim(sim)

    def fail_affine_transform(*args, **kwargs):
        raise AssertionError("no-op transform should not resample")

    monkeypatch.setattr(
        transformation,
        "dask_image_affine_transform",
        fail_affine_transform,
    )

    transformed = transformation.transform_sim(
        sim,
        p=param_utils.identity_transform(2),
        output_stack_properties=output_stack_properties,
        order=3,
    )

    assert transformed.data is sim.data
    np.testing.assert_array_equal(transformed.compute().data, data[None, None])
    assert si_utils.get_stack_properties_from_sim(transformed) == (
        output_stack_properties
    )


def test_transform_sim_centers_large_origins_and_preserves_small_scale(
    monkeypatch,
):
    origin = 1e12
    scale_change = 1e-8
    sim = si_utils.get_sim_from_array(
        np.ones((2, 2), dtype=np.float32),
        dims=["y", "x"],
        scale={"y": 1.0, "x": 1.0},
        translation={"y": origin, "x": origin},
    ).compute()
    output_stack_properties = si_utils.get_stack_properties_from_sim(sim)
    affine = param_utils.identity_transform(2).values
    affine[1, 1] += scale_change
    affine[1, 2] = -10000.0

    captured = {}

    def capture_affine_transform(data, **kwargs):
        captured.update(kwargs)
        return np.asarray(data)

    monkeypatch.setattr(
        transformation,
        "affine_transform",
        capture_affine_transform,
    )

    transformation.transform_sim(
        sim,
        p=affine,
        output_stack_properties=output_stack_properties,
        order=0,
    )

    # The near-identity scale is genuine and must not be snapped to one.
    assert captured["matrix"][1, 1] == 1.0 + scale_change

    # Centered arithmetic retains the small residual encoded by the affine;
    # evaluating A * origin + offset - origin directly rounds it to zero.
    expected_offset = (affine[1, 1] - 1.0) * origin - 10000.0
    direct_offset = affine[1, 1] * origin - 10000.0 - origin
    assert direct_offset == 0.0
    assert expected_offset != 0.0
    np.testing.assert_allclose(
        captured["offset"][1], expected_offset, atol=1e-8
    )
