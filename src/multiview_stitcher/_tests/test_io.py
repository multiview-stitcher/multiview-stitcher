import os
import tempfile

import numpy as np
import pytest

from multiview_stitcher import io, msi_utils, sample_data


@pytest.mark.parametrize(
    "ndim, N_t, N_c",
    [(ndim, N_t, N_c) for ndim in [2, 3] for N_t in [1, 2] for N_c in [1, 2]],
)
def test_tiff_io(ndim, N_t, N_c):
    """
    Could be much more general
    """

    tile_size = 10
    spacing_x = 0.5
    spacing_y = 0.5
    spacing_z = 0.5
    sims = sample_data.generate_tiled_dataset(
        ndim=ndim,
        overlap=0,
        N_c=N_c,
        N_t=N_t,
        tile_size=tile_size,
        tiles_x=1,
        tiles_y=1,
        tiles_z=1,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        spacing_z=spacing_z,
        drift_scale=0,
        shift_scale=0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.tif")
        io.save_sim_as_tif(filepath, sims[0])

        sims_io = io.read_tiff_into_spatial_xarray(
            filepath, channel_names=["ch%s" % i for i in range(N_c)]
        )

        assert sims[0].data.ndim == sims_io.data.ndim

        # check that all dims have the same length
        for dim in sims[0].dims:
            assert len(sims[0].coords[dim]) == len(sims_io.coords[dim])
            # assert np.allclose(sims[0].coords[dim], sims_io.coords[dim])

        # check image values are the same
        # ignore coordinates for this test
        for dim in sims[0].dims:
            sims[0].coords[dim] = np.arange(len(sims[0].coords[dim]))
            sims_io.coords[dim] = np.arange(len(sims_io.coords[dim]))

        assert (sims[0] == sims_io).min()


def test_read_imaris_into_msim_synthetic_file(tmp_path):
    h5py = pytest.importorskip("h5py")
    filepath = tmp_path / "synthetic.ims"
    data0 = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)

    def _ims_attr(value):
        return np.frombuffer(f"{value}\x00".encode("utf-8"), dtype="S1")


    def _require_group(root, path):
        group = root
        for part in path.split("/"):
            group = group.require_group(part)
        return group

    with h5py.File(filepath, "w") as f:
        image_info = _require_group(f, "DataSetInfo/Image")
        for name, value in {
            "X": 4,
            "Y": 3,
            "Z": 2,
            "ExtMin0": 0.0,
            "ExtMin1": 0.0,
            "ExtMin2": 0.0,
            "ExtMax0": 4.0,
            "ExtMax1": 6.0,
            "ExtMax2": 6.0,
        }.items():
            image_info.attrs[name] = _ims_attr(value)

        for ires, shape in [(0, (2, 3, 4)), (1, (1, 2, 2))]:
            for ichannel in [0, 1]:
                group = _require_group(
                    f,
                    "DataSet/"
                    f"ResolutionLevel {ires}/TimePoint 0/Channel {ichannel}",
                )
                group.attrs["ImageSizeZ"] = _ims_attr(shape[0])
                group.attrs["ImageSizeY"] = _ims_attr(shape[1])
                group.attrs["ImageSizeX"] = _ims_attr(shape[2])

                if ires == 0:
                    data = np.full((2, 4, 4), 999, dtype=np.uint16)
                    data[:, :3, :] = data0 + ichannel * 100
                else:
                    data = np.full(shape, ichannel + 10, dtype=np.uint16)
                group.create_dataset("Data", data=data, chunks=(1, 1, 2))

    msim = io.read_imaris_into_msim(filepath, channels=[1])
    sim0 = msi_utils.get_sim_from_msim(msim, scale="scale0")

    assert msi_utils.get_sorted_scale_keys(msim) == ["scale0", "scale1"]
    assert sim0.dims == ("t", "c", "z", "y", "x")
    assert sim0.shape == (1, 1, 2, 3, 4)
    assert sim0.coords["c"].values.tolist() == [1]
    assert sim0.coords["t"].values.tolist() == [0]
    np.testing.assert_allclose(sim0.coords["z"].values, [0.0, 3.0])
    np.testing.assert_allclose(sim0.coords["y"].values, [0.0, 2.0, 4.0])
    np.testing.assert_allclose(sim0.coords["x"].values, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        sim0.sel(t=0, c=1).data.compute(),
        data0 + 100,
    )
