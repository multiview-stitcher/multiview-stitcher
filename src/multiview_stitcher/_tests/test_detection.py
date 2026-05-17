import dask.array as da
import numpy as np
from scipy.ndimage import gaussian_filter

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

    mask = detection.log_detect(
        image,
        target_size=4.0,
        threshold_rel=0.3,
    )
    detected = np.argwhere(mask)
    detected = detected[np.argsort(detected[:, 0])]

    assert mask.dtype == bool
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

    positions = detection.detect_beads(
        msim,
        target_size_physical=2.0,
        threshold_rel=0.3,
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
    def threshold_detect(image, target_size, threshold):
        return image > threshold

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
        target_size_physical=6.0,
        detection_func=threshold_detect,
        detection_func_kwargs={"threshold": 1.0},
        detection_overlap=0,
        segmentation_res_level=0,
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
