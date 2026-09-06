"""Physical-coordinate and pixel-preservation contracts for OME-Zarr 0.6."""

from copy import deepcopy
from functools import partial
import hashlib

import dask.array as da
from dask.callbacks import Callback
import ngff_zarr
import numpy as np
import pytest
import zarr

from multiview_stitcher import msi_utils, ngff_utils, param_utils
from multiview_stitcher import spatial_image_utils as si_utils


def make_sim(ndim=2, n_t=1):
    dims = ["z", "y", "x"][-ndim:]
    shape = (n_t, 2) + (16,) * ndim
    sim = si_utils.get_sim_from_array(
        da.from_array(
            np.arange(np.prod(shape), dtype="uint16").reshape(shape),
            chunks=(1, 1) + (8,) * ndim,
        ),
        dims=["t", "c"] + dims,
        scale=dict(zip(dims, [2.5, 0.7, 0.3][-ndim:])),
        translation=dict(zip(dims, [11.0, -4.0, 7.0][-ndim:])),
        c_coords=["DAPI", "GFP"],
        transform_key="registered",
    )
    affine = np.eye(ndim + 1)
    affine[:2, :2] = [[0.0, -1.0], [1.0, 0.2]]  # rotation and shear
    affine[:-1, -1] = np.arange(ndim) + 13.0
    si_utils.set_sim_affine(
        sim,
        param_utils.affine_to_xaffine(affine, t_coords=sim.coords["t"].values),
        "registered",
    )
    ngff_utils.set_ngff_time_transform(
        sim, {"scale": 3.0, "translation": 2.0, "unit": "second"}
    )
    return sim, affine


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("backend", ["zarr", "dask"])
def test_affine_roundtrip_and_pyramid_landmarks(
    tmp_path, ndim, backend, monkeypatch
):
    monkeypatch.setattr(
        msi_utils,
        "calc_resolution_levels",
        partial(msi_utils.calc_resolution_levels, min_shape=3),
    )
    sim, affine = make_sim(ndim)
    path = tmp_path / "tile.zarr"
    ngff_utils.write_sim_to_ome_zarr(
        sim,
        path,
        ngff_version="0.6",
        transform_key="registered",
        downscale_factors_per_spatial_dim={
            d: 2 for d in si_utils.get_spatial_dims_from_sim(sim)
        },
    )
    executed = []
    with Callback(pretask=lambda *args: executed.append(args)):
        msim = ngff_utils.read_msim_from_ome_zarr(path, array_backend=backend)
    assert not executed, "Reading metadata must not execute pixel tasks"
    assert len(msi_utils.get_sorted_scale_keys(msim)) == 3
    metadata = zarr.open_group(path).attrs["ome"]
    assert metadata["version"] == "0.6.dev4"
    from ngff_zarr.validate import validate

    validate({"ome": metadata}, version="0.6.dev4")
    assert metadata["omero"]["channels"][0]["label"] == "DAPI"
    spacing = np.array(list(si_utils.get_spacing_from_sim(sim).values()))
    origin = np.array(list(si_utils.get_origin_from_sim(sim).values()))
    sdims = si_utils.get_spatial_dims_from_sim(sim)
    for level, key in enumerate(msi_utils.get_sorted_scale_keys(msim)):
        restored = msi_utils.get_sim_from_msim(msim, scale=key)
        matrix = si_utils.get_affine_from_sim(
            restored, si_utils.DEFAULT_TRANSFORM_KEY
        ).values[0]
        np.testing.assert_allclose(matrix, affine)
        factor = 2**level
        pixel = np.arange(ndim) + 1.0
        expected_intrinsic = origin + spacing * (
            (factor - 1) / 2 + factor * pixel
        )
        actual_intrinsic = np.array(
            [
                float(restored.coords[d][0])
                + float(restored.coords[d][1] - restored.coords[d][0])
                * pixel[i]
                for i, d in enumerate(sdims)
            ]
        )
        np.testing.assert_allclose(
            matrix @ np.r_[actual_intrinsic, 1.0],
            affine @ np.r_[expected_intrinsic, 1.0],
        )
        assert ngff_utils.get_ngff_time_transform(restored) == {
            "scale": 3.0,
            "translation": 2.0,
            "unit": "second",
        }
        assert list(restored.c.values) == ["DAPI", "GFP"]
    restored = ngff_utils.read_sim_from_ome_zarr(path, array_backend=backend)
    np.testing.assert_array_equal(
        np.asarray(restored.data), np.asarray(sim.data)
    )
    intrinsic = ngff_utils.read_sim_from_ome_zarr(
        path, target_coordinate_system="intrinsic"
    )
    np.testing.assert_allclose(
        si_utils.get_affine_from_sim(
            intrinsic, si_utils.DEFAULT_TRANSFORM_KEY
        ).values[0],
        np.eye(ndim + 1),
    )


def test_in_memory_multiscales_roundtrip(tmp_path):
    sim, affine = make_sim()
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[2])
    ngff = ngff_utils.msim_to_ngff_multiscales(
        msim, "registered", ngff_version="0.6"
    )
    path = tmp_path / "image.zarr"
    ngff_zarr.to_ngff_zarr(path, ngff, version="0.6")
    restored = ngff_utils.read_msim_from_ome_zarr(path)
    for key in msi_utils.get_sorted_scale_keys(restored):
        actual = msi_utils.get_sim_from_msim(restored, scale=key)
        expected = msi_utils.get_sim_from_msim(msim, scale=key)
        np.testing.assert_array_equal(
            np.asarray(actual.data), np.asarray(expected.data)
        )
        np.testing.assert_allclose(
            si_utils.get_affine_from_sim(
                actual, si_utils.DEFAULT_TRANSFORM_KEY
            ).values[0],
            affine,
        )


def test_registration_update_preserves_pixels_and_calibration(tmp_path):
    sim, affine = make_sim()
    path = tmp_path / "tile.zarr"
    ngff_utils.write_sim_to_ome_zarr(
        sim, path, ngff_version="0.6", transform_key="registered"
    )
    root = zarr.open_group(path, mode="a")
    ome = dict(root.attrs["ome"])
    ome["custom"] = {"keep": True}
    ome["multiscales"][0]["metadata"] = {"method": "keep"}
    root.attrs["ome"] = ome
    before = deepcopy(ome)
    chunk_hashes = {
        p.relative_to(path): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob("*")
        if p.is_file() and p.name != "zarr.json"
    }
    msim = ngff_utils.read_msim_from_ome_zarr(path, transform_key="registered")
    affine[0, -1] += 19
    msi_utils.set_affine_transform(
        msim,
        param_utils.affine_to_xaffine(affine, t_coords=[0]),
        transform_key="registered",
    )
    ngff_utils.update_ome_zarr_multiscales_metadata(path, msim, "registered")
    after = zarr.open_group(path).attrs["ome"]
    assert (
        before["multiscales"][0]["datasets"]
        == after["multiscales"][0]["datasets"]
    )
    assert before["omero"] == after["omero"]
    assert before["custom"] == after["custom"]
    assert (
        before["multiscales"][0]["metadata"]
        == after["multiscales"][0]["metadata"]
    )
    assert chunk_hashes == {
        p.relative_to(path): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob("*")
        if p.is_file() and p.name != "zarr.json"
    }
    actual = ngff_utils.read_sim_from_ome_zarr(path)
    np.testing.assert_allclose(
        si_utils.get_affine_from_sim(
            actual, si_utils.DEFAULT_TRANSFORM_KEY
        ).values[0],
        affine,
    )


def write_modified_metadata(path, change):
    root = zarr.open_group(path, mode="a")
    ome = dict(root.attrs["ome"])
    change(ome["multiscales"][0])
    root.attrs["ome"] = ome


def test_ambiguous_targets_require_selection(tmp_path):
    sim, affine = make_sim()
    ngff_utils.write_sim_to_ome_zarr(
        sim, tmp_path, ngff_version="0.6", transform_key="registered"
    )

    def add_target(entry):
        cs = deepcopy(entry["coordinateSystems"][1])
        cs["name"] = "atlas"
        tf = deepcopy(entry["coordinateTransformations"][0])
        tf["output"]["name"] = "atlas"
        entry["coordinateSystems"].append(cs)
        entry["coordinateTransformations"].append(tf)

    write_modified_metadata(tmp_path, add_target)
    with pytest.raises(ValueError, match="Multiple registrations"):
        ngff_utils.read_sim_from_ome_zarr(tmp_path)
    actual = ngff_utils.read_sim_from_ome_zarr(
        tmp_path, target_coordinate_system="atlas"
    )
    np.testing.assert_allclose(
        si_utils.get_affine_from_sim(
            actual, si_utils.DEFAULT_TRANSFORM_KEY
        ).values[0],
        affine,
    )


@pytest.mark.parametrize("kind", ["displacements", "mixing", "calibration"])
def test_unsupported_transforms_are_not_silently_dropped(tmp_path, kind):
    sim, _ = make_sim()
    ngff_utils.write_sim_to_ome_zarr(
        sim, tmp_path, ngff_version="0.6", transform_key="registered"
    )

    def change(entry):
        tf = entry["coordinateTransformations"][0]
        if kind == "displacements":
            tf.pop("affine")
            tf.update(type="displacements", path="field")
        elif kind == "mixing":
            tf["affine"][2][0] = 1.0
        else:
            entry["datasets"][0]["coordinateTransformations"][0][
                "transformations"
            ].reverse()

    write_modified_metadata(tmp_path, change)
    with pytest.raises((NotImplementedError, ValueError)):
        ngff_utils.read_sim_from_ome_zarr(tmp_path)


def test_reject_varying_affine_before_overwrite(tmp_path):
    sim, _ = make_sim(n_t=2)
    ngff_utils.write_sim_to_ome_zarr(sim, tmp_path, ngff_version="0.6")
    before = (tmp_path / "zarr.json").read_bytes()
    sim.attrs["transforms"]["registered"].values[1, 0, -1] += 1
    with pytest.raises(NotImplementedError, match="static registration"):
        ngff_utils.write_sim_to_ome_zarr(
            sim,
            tmp_path,
            ngff_version="0.6",
            transform_key="registered",
            overwrite=True,
        )
    assert (tmp_path / "zarr.json").read_bytes() == before


def test_sequence_composition_order(tmp_path):
    sim, _ = make_sim()
    ngff_utils.write_sim_to_ome_zarr(
        sim, tmp_path, ngff_version="0.6", transform_key="registered"
    )

    def change(entry):
        tf = entry["coordinateTransformations"][0]
        tf.pop("affine")
        tf.update(
            type="sequence",
            transformations=[
                {"type": "translation", "translation": [0, 0, 2, 3]},
                {"type": "scale", "scale": [1, 1, 4, 5]},
            ],
        )

    write_modified_metadata(tmp_path, change)
    actual = ngff_utils.read_sim_from_ome_zarr(tmp_path)
    np.testing.assert_allclose(
        si_utils.get_affine_from_sim(
            actual, si_utils.DEFAULT_TRANSFORM_KEY
        ).values[0],
        [[4, 0, 8], [0, 5, 15], [0, 0, 1]],
    )


def test_fusion_roundtrip_matches_in_memory(tmp_path):
    from multiview_stitcher import fusion

    sim, _ = make_sim()
    path = tmp_path / "input.zarr"
    ngff_utils.write_sim_to_ome_zarr(
        sim, path, ngff_version="0.6", transform_key="registered"
    )
    restored = ngff_utils.read_sim_from_ome_zarr(
        path, transform_key="registered"
    )
    other, other_affine = make_sim()
    other_affine[1, -1] += 4
    si_utils.set_sim_affine(
        other,
        param_utils.affine_to_xaffine(other_affine, t_coords=[0]),
        "registered",
    )
    ngff_utils.write_sim_to_ome_zarr(
        other,
        tmp_path / "other.zarr",
        ngff_version="0.6",
        transform_key="registered",
    )
    other_restored = ngff_utils.read_sim_from_ome_zarr(
        tmp_path / "other.zarr", transform_key="registered"
    )
    expected = fusion.fuse(
        [sim, other], transform_key="registered", output_chunksize=64
    )
    actual = fusion.fuse(
        [restored, other_restored],
        transform_key="registered",
        output_chunksize=64,
        output_zarr_url=str(tmp_path / "fused.zarr"),
        zarr_options={"ome_zarr": True, "ngff_version": "0.6"},
    )
    np.testing.assert_allclose(
        np.asarray(actual.data), np.asarray(expected.data)
    )
