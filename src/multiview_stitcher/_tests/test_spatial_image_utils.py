import os
import tempfile

import dask.array as da
import numpy as np
import pytest
import xarray as xr
import zarr

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


def test_sim_zarr_array_input_backend_is_preserved():
    with tempfile.TemporaryDirectory() as tmpdir:
        zarray = zarr.open_array(
            os.path.join(tmpdir, "input.zarr"),
            mode="w",
            shape=(8, 8),
            chunks=(4, 4),
            dtype=np.uint16,
        )
        zarray[:] = 1

        sim = si_utils.get_sim_from_array(
            zarray,
            dims=["y", "x"],
            scale={"y": 1.0, "x": 1.0},
            translation={"y": 0.0, "x": 0.0},
        )

        assert si_utils.is_xarray_zarr_backed(sim)

        sim_slice = sim.sel(y=slice(0.0, 3.0), x=slice(0.0, 3.0))
        sim_slice = si_utils.ensure_dask_backed_dataarray(sim_slice)

        assert isinstance(sim_slice.data, da.Array)


def test_sim_zarr_array_html_repr_reuses_zarr_repr():
    with tempfile.TemporaryDirectory() as tmpdir:
        zarray = zarr.open_array(
            os.path.join(tmpdir, "input.zarr"),
            mode="w",
            shape=(8, 8),
            chunks=(4, 4),
            dtype=np.uint16,
        )
        zarray[:] = 1

        sim = si_utils.get_sim_from_array(
            zarray,
            dims=["y", "x"],
            scale={"y": 1.0, "x": 1.0},
            translation={"y": 0.0, "x": 0.0},
        )

        assert hasattr(sim.variable._data, "_repr_html_")
        assert hasattr(sim.variable._data, "_repr_inline_")
        html_repr = sim.variable._data._repr_html_().lower()
        inline_repr = sim.variable._data._repr_inline_(120).lower()

        assert "zarr-backed; using dask-style repr." in html_repr
        assert "dask" in html_repr
        assert "zarr-backed; using dask-style repr." in inline_repr
        assert "dask" in inline_repr


def test_serialize_deserialize_zarr_backed_sim_roundtrip():
    """Roundtrip: serialize then deserialize preserves dims and zarr-backed status."""
    with tempfile.TemporaryDirectory() as tmpdir:
        zarray = zarr.open_array(
            os.path.join(tmpdir, "input.zarr"),
            mode="w",
            shape=(3, 2, 8, 8),
            chunks=(1, 1, 4, 4),
            dtype=np.uint16,
        )
        zarray[:] = 1

        sim = si_utils.get_sim_from_array(
            zarray,
            dims=["t", "c", "y", "x"],
            scale={"y": 1.0, "x": 1.0},
            translation={"y": 0.0, "x": 0.0},
            t_coords=[0.0, 1.0, 2.0],
            c_coords=[0, 1],
        )

        info = si_utils.serialize_zarr_backed_sim(sim)
        sim2 = si_utils.deserialize_zarr_backed_sim(info)

        assert list(sim2.dims) == list(sim.dims)
        assert si_utils.is_xarray_zarr_backed(sim2)


def test_serialize_deserialize_zarr_backed_sim_with_dropped_dim():
    """Roundtrip after sim_sel_coords({'t': t_val}) preserves zarr-backed status."""
    with tempfile.TemporaryDirectory() as tmpdir:
        zarray = zarr.open_array(
            os.path.join(tmpdir, "input.zarr"),
            mode="w",
            shape=(3, 2, 8, 8),
            chunks=(1, 1, 4, 4),
            dtype=np.uint16,
        )
        zarray[:] = 1

        sim = si_utils.get_sim_from_array(
            zarray,
            dims=["c", "t", "y", "x"],
            scale={"y": 1.0, "x": 1.0},
            translation={"y": 0.0, "x": 0.0},
            t_coords=[10.0, 20.0],
            c_coords=[0, 1, 2],
        )

        # Simulate the sim_sel_coords({'t': 10.0}) pattern (no drop=True)
        sim_sel = sim.sel({"t": 10.0})

        assert "t" not in sim_sel.dims

        info = si_utils.serialize_zarr_backed_sim(sim_sel)
        sim2 = si_utils.deserialize_zarr_backed_sim(info)

        assert info["zarr_dims"] == ["c", "t", "y", "x"]
        assert info["isel_dropped"] == {"t": 0}
        assert list(sim2.dims) == list(sim_sel.dims)
        assert si_utils.is_xarray_zarr_backed(sim2)
        # Spatial coords are preserved
        assert sim2.coords["y"].values == pytest.approx(sim_sel.coords["y"].values)
        assert sim2.coords["x"].values == pytest.approx(sim_sel.coords["x"].values)


def test_serialize_deserialize_zarr_backed_sim_after_singleton_expansion():
    """Roundtrip works when zarr dims are a subset of the expanded sim dims."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sim = si_utils.get_sim_from_array(
            np.ones((1, 8, 8), dtype=np.uint16),
            dims=["c", "y", "x"],
            scale={"y": 1.0, "x": 1.0},
            translation={"y": 0.0, "x": 0.0},
        )
        sim = si_utils.sim_sel_coords(sim, {"t": 0})

        zarray = zarr.open_array(
            os.path.join(tmpdir, "input.zarr"),
            mode="w",
            shape=sim.shape,
            chunks=(1, 4, 4),
            dtype=np.uint16,
        )
        zarray[:] = 1

        sim = si_utils.get_sim_from_array(
            zarray,
            dims=sim.dims,
            scale=si_utils.get_spacing_from_sim(sim),
            translation=si_utils.get_origin_from_sim(sim),
        )
        sim_sel = si_utils.sim_sel_coords(sim, {"t": 0, "c": 0})

        info = si_utils.serialize_zarr_backed_sim(sim_sel)
        sim2 = si_utils.deserialize_zarr_backed_sim(info)

        assert info["zarr_dims"] == ["c", "y", "x"]
        assert info["isel_dropped"] == {"t": 0, "c": 0}
        assert list(sim2.dims) == list(sim_sel.dims)
        assert si_utils.is_xarray_zarr_backed(sim2)


def test_deserialize_zarr_backed_sim_reconstruct_slice_reads_requested_region():
    with tempfile.TemporaryDirectory() as tmpdir:
        data = np.arange(3 * 2 * 8 * 8, dtype=np.uint16).reshape(3, 2, 8, 8)
        zarray = zarr.open_array(
            os.path.join(tmpdir, "input.zarr"),
            mode="w",
            shape=data.shape,
            chunks=(1, 1, 4, 4),
            dtype=data.dtype,
        )
        zarray[:] = data

        sim = si_utils.get_sim_from_array(
            zarray,
            dims=["t", "c", "y", "x"],
            scale={"y": 1.0, "x": 1.0},
            translation={"y": 10.0, "x": 20.0},
            t_coords=[0.0, 1.0, 2.0],
            c_coords=[10, 20],
        )
        sim_sel = sim.sel({"t": 1.0})

        helper_calls = []
        original_materialize = si_utils._materialize_xarray_zarr_backend

        def record_materialize(xim, *args, **kwargs):
            helper_calls.append(
                {
                    "dims": list(xim.dims),
                    "is_zarr_backed": si_utils.is_xarray_zarr_backed(xim),
                }
            )
            return original_materialize(xim, *args, **kwargs)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            si_utils,
            "_materialize_xarray_zarr_backend",
            record_materialize,
        )

        try:
            sim2 = si_utils.deserialize_zarr_backed_sim(
                si_utils.serialize_zarr_backed_sim(sim_sel),
                reconstruct_slice=True,
                overlap_bb={
                    "origin": {"y": 12.0, "x": 23.0},
                    "shape": {"y": 3, "x": 2},
                    "spacing": {"y": 1.0, "x": 1.0},
                },
                sim_coord_dict={"c": 20},
            )
        finally:
            monkeypatch.undo()

        assert list(sim2.dims) == ["y", "x"]
        assert not si_utils.is_xarray_zarr_backed(sim2)
        assert not si_utils.is_dask_backed_dataarray(sim2)
        assert helper_calls == [{"dims": ["y", "x"], "is_zarr_backed": True}]
        np.testing.assert_array_equal(np.asarray(sim2.data), data[1, 1, 2:5, 3:5])
        assert sim2.coords["y"].values == pytest.approx([12.0, 13.0, 14.0])
        assert sim2.coords["x"].values == pytest.approx([23.0, 24.0])


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


@pytest.mark.parametrize("ndim", [2, 3])
def test_get_extent_from_sim(ndim):
    shape = {dim: 10 for dim in si_utils.SPATIAL_DIMS[-ndim:]}
    scale = {dim: 0.5 for dim in si_utils.SPATIAL_DIMS[-ndim:]}

    sim = si_utils.get_sim_from_array(
        np.ones(tuple(shape.values())),
        dims=list(shape.keys()),
        scale=scale,
        translation={dim: 0.0 for dim in shape},
    )

    extent = si_utils.get_extent_from_sim(sim)

    for dim in shape:
        assert extent[dim] == pytest.approx(scale[dim] * (shape[dim] - 1))


def test_point_set_helpers_and_spatial_selection():
    sim = si_utils.get_sim_from_array(
        np.zeros((5, 5)),
        dims=["y", "x"],
        scale={"y": 1.0, "x": 2.0},
        translation={"y": 10.0, "x": 20.0},
    )

    points = np.array(
        [
            [10.0, 20.0],
            [12.0, 24.0],
            [14.0, 28.0],
        ]
    )
    si_utils.set_point_set(sim, points)

    point_set = si_utils.get_point_set(sim)
    assert point_set["position"].dims == ("t", "c", "point", "dim")
    assert np.allclose(point_set["position"].isel(t=0, c=0), points)

    selected = si_utils.sim_sel_coords(
        sim,
        {"y": slice(10.0, 12.0), "x": slice(20.0, 24.0)},
    )
    selected_points = si_utils.get_point_set(selected)

    assert selected_points["position"].dims == ("t", "c", "point", "dim")
    assert np.allclose(
        selected_points["position"].isel(t=0, c=0).values,
        np.array([[10.0, 20.0], [12.0, 24.0]]),
    )


def test_point_set_nonspatial_selection():
    sim = si_utils.get_sim_from_array(
        np.zeros((2, 5, 5)),
        dims=["t", "y", "x"],
        t_coords=[0, 1],
    )
    points = xr.Dataset(
        {
            "position": xr.DataArray(
                np.array(
                    [
                        [[1.0, 1.0], [2.0, 2.0]],
                        [[3.0, 3.0], [4.0, 4.0]],
                    ]
                ),
                dims=["t", "point", "dim"],
                coords={"t": [0, 1], "dim": ["y", "x"]},
            )
        }
    )
    si_utils.set_point_set(sim, points)

    selected = si_utils.sim_sel_coords(sim, {"t": 1})
    selected_points = si_utils.get_point_set(selected)

    assert "t" not in selected_points["position"].dims
    assert selected_points["position"].dims == ("c", "point", "dim")
    assert np.allclose(
        selected_points["position"].isel(c=0).values,
        np.array([[3.0, 3.0], [4.0, 4.0]]),
    )


def test_point_set_spatial_selection_preserves_nonspatial_dims():
    sim = si_utils.get_sim_from_array(
        np.zeros((2, 5, 5)),
        dims=["t", "y", "x"],
        t_coords=[0, 1],
    )
    points = xr.Dataset(
        {
            "position": xr.DataArray(
                np.array(
                    [
                        [[1.0, 1.0], [4.0, 4.0]],
                        [[4.0, 4.0], [1.0, 1.0]],
                    ]
                ),
                dims=["t", "point", "dim"],
                coords={"t": [0, 1], "dim": ["y", "x"]},
            )
        }
    )
    si_utils.set_point_set(sim, points)

    selected = si_utils.sim_sel_coords(
        sim,
        {"y": slice(0.0, 2.0), "x": slice(0.0, 2.0)},
    )
    selected_points = si_utils.get_point_set(selected)

    assert selected_points["position"].dims == ("t", "c", "point", "dim")
    assert np.allclose(
        selected_points["position"].isel(c=0).values,
        np.array(
            [
                [[1.0, 1.0], [np.nan, np.nan]],
                [[np.nan, np.nan], [1.0, 1.0]],
            ]
        ),
        equal_nan=True,
    )
