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
