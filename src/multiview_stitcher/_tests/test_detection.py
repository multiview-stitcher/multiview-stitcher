import dask
import dask.array as da
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, label

from multiview_stitcher import detection, msi_utils
from multiview_stitcher import spatial_image_utils as si_utils


def _make_bead_image(bead_pixels):
    image = np.zeros((64, 64), dtype=np.float32)
    image[tuple(bead_pixels.T)] = [100.0, 80.0]
    return gaussian_filter(image, sigma=1.4)


def test_log_detect_detects_numpy_beads():
    bead_pixels = np.array(
        [
            [20, 22],
            [43, 41],
        ]
    )
    image = _make_bead_image(bead_pixels)

    labels = detection.log_detect(
        image,
        spacing=(1.0, 1.0),
        target_size_physical=4,
        threshold_rel=0.3,
    )
    detected = np.argwhere(labels > 0)
    detected = detected[np.argsort(detected[:, 0])]

    assert np.issubdtype(labels.dtype, np.integer)
    assert detected.shape == bead_pixels.shape
    assert np.allclose(detected, bead_pixels, atol=1)


def test_detect_beads_returns_intrinsic_physical_positions():
    bead_pixels = np.array(
        [
            [20, 22],
            [43, 41],
        ]
    )
    image = _make_bead_image(bead_pixels)

    spacing = {"y": 0.5, "x": 0.5}
    origin = {"y": 5.0, "x": -2.0}
    sim = si_utils.get_sim_from_array(
        da.from_array(image, chunks=(16, 16)),
        dims=["y", "x"],
        scale=spacing,
        translation=origin,
    )
    msim = msi_utils.get_msim_from_sim(
        sim,
        scale_factors=[{"y": 2, "x": 2}],
    )

    with dask.config.set(scheduler="synchronous"):
        positions = detection.detect_beads(
            msim,
            detection_func_kwargs={
                "target_size_physical": 2,
                "threshold_rel": 0.3,
            },
        )

    expected = np.column_stack(
        [
            origin[dim] + bead_pixels[:, idim] * spacing[dim]
            for idim, dim in enumerate(["y", "x"])
        ]
    )
    detected = positions.values[np.argsort(positions.values[:, 0])]

    assert positions.dims == ("point_id", "dim")
    assert list(positions.coords["dim"].values) == ["y", "x"]
    assert positions.attrs["segmentation_scale"] == "scale0"
    assert detected.shape == expected.shape
    assert np.allclose(detected, expected, atol=max(spacing.values()))


def test_detect_beads_accepts_custom_detection_func():
    def threshold_detect(image, spacing, threshold):
        return label(image > threshold)[0]

    bead_pixels = np.array(
        [
            [11, 13],
            [42, 38],
        ]
    )
    image = np.zeros((64, 64), dtype=np.float32)
    image[tuple(bead_pixels.T)] = 5.0

    spacing = {"y": 2.0, "x": 3.0}
    origin = {"y": -4.0, "x": 7.0}
    sim = si_utils.get_sim_from_array(
        da.from_array(image, chunks=(16, 16)),
        dims=["y", "x"],
        scale=spacing,
        translation=origin,
    )
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])

    positions = detection.detect_beads(
        msim,
        detection_func=threshold_detect,
        detection_func_kwargs={"threshold": 1.0},
        detection_overlap=0,
    )

    expected = np.column_stack(
        [
            origin[dim] + bead_pixels[:, idim] * spacing[dim]
            for idim, dim in enumerate(["y", "x"])
        ]
    )
    detected = positions.values[np.argsort(positions.values[:, 0])]

    assert positions.attrs["detection_func"] == "threshold_detect"
    assert detected.shape == expected.shape
    assert np.allclose(detected, expected)


def test_detect_beads_keeps_chunk_boundary_label_once():
    def threshold_detect(image, spacing, threshold):
        return label(image > threshold)[0]

    image = np.zeros((32, 32), dtype=np.float32)
    image[14:18, 10:14] = 5.0

    sim = si_utils.get_sim_from_array(
        da.from_array(image, chunks=(16, 16)),
        dims=["y", "x"],
        scale={"y": 1.0, "x": 1.0},
        translation={"y": 0.0, "x": 0.0},
    )
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])

    positions = detection.detect_beads(
        msim,
        detection_func=threshold_detect,
        detection_func_kwargs={"threshold": 1.0},
        detection_overlap=4,
    )

    assert positions.shape == (1, 2)
    assert np.allclose(positions.values[0], [15.5, 11.5])


def test_detect_beads_uses_max_detection_spacing_for_scale_selection():
    def empty_detect(image, spacing):
        return np.zeros_like(image, dtype=np.int64)

    sim = si_utils.get_sim_from_array(
        da.from_array(np.zeros((32, 32), dtype=np.float32), chunks=(16, 16)),
        dims=["y", "x"],
        scale={"y": 1.0, "x": 1.0},
        translation={"y": 0.0, "x": 0.0},
    )
    msim = msi_utils.get_msim_from_sim(
        sim,
        scale_factors=[{"y": 2, "x": 2}],
    )

    positions = detection.detect_beads(
        msim,
        detection_func=empty_detect,
        max_detection_spacing={"y": 2.0, "x": 2.0},
    )

    assert positions.attrs["segmentation_scale"] == "scale1"
    assert positions.shape == (0, 2)


@pytest.mark.parametrize(
    "backend, expected",
    [(None, "numpy"), ("cupy", "cupy")],
)
def test_detect_beads_passes_backend_to_fuse(monkeypatch, backend, expected):
    seen = {}

    def fake_fuse(*args, **kwargs):
        seen["backend"] = kwargs["backend"]
        labels = np.zeros((4, 4), dtype=np.int64)
        labels[1, 2] = 1
        return type(
            "FusedLabels",
            (),
            {"data": da.from_array(labels, chunks=labels.shape)},
        )()

    def empty_detect(image, spacing):
        return np.zeros_like(image, dtype=np.int64)

    sim = si_utils.get_sim_from_array(
        np.zeros((4, 4), dtype=np.float32),
        dims=["y", "x"],
        scale={"y": 1.0, "x": 1.0},
        translation={"y": 0.0, "x": 0.0},
    )
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])

    monkeypatch.setattr(detection.fusion, "fuse", fake_fuse)

    kwargs = {} if backend is None else {"backend": backend}
    positions = detection.detect_beads(
        msim,
        detection_func=empty_detect,
        detection_overlap=0,
        **kwargs,
    )

    assert seen["backend"] == expected
    assert np.allclose(positions.values, [[1.0, 2.0]])
