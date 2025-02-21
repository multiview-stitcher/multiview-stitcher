from pathlib import Path

import pytest
import xarray as xr

import multiview_stitcher.spatial_image_utils as si_utils
from multiview_stitcher import (
    io,
    msi_utils,
    registration,
)
from multiview_stitcher.io import METADATA_TRANSFORM_KEY

test_bench_data_dir = (
    Path(__file__).parent.parent.parent.parent
    / "image-datasets"
    / "test_bench_data"
)

datasets = [
    {
        "name": "tcyx_2x1_0",
        "image_paths": [test_bench_data_dir / "tcyx_2x1_0.czi"],
        "parameter_path": test_bench_data_dir / "tcyx_2x1_0.json",
        "dims": ["t", "c", "y", "x"],
        "visibility": "private",
    },
    {
        "name": "tcyx_3x3_0",
        "image_paths": [test_bench_data_dir / "tcyx_3x3_0.czi"],
        "parameter_path": test_bench_data_dir / "tcyx_3x3_0.json",
        "dims": ["t", "c", "y", "x"],
        "visibility": "private",
    },
]


def read_params(dataset):
    xparamss = xr.open_zarr(dataset["parameter_path"]).compute()

    return [xparamss[str(ind)] for ind in range(len(xparamss))]


def write_params(params, dataset):
    xparams = xr.Dataset(
        {str(ind): param for ind, param in enumerate(params)},
    )

    xparams.to_zarr(dataset["parameter_path"])


def get_msims_from_dataset(dataset):
    if dataset["image_paths"][0].suffix == ".czi":
        sims = io.read_mosaic_image_into_list_of_spatial_xarrays(
            dataset["image_paths"][0]
        )

    msims = [
        msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims
    ]

    return msims


def register_dataset(msims, dataset):
    si_utils.get_spatial_dims_from_sim(msi_utils.get_sim_from_msim(msims[0]))

    params = registration.register(
        msims,
        # registration_binning={dim: 1 for dim in sdims},
        reg_channel_index=0
        if "reg_channel_index" not in dataset
        else dataset["reg_channel_index"],
        transform_key=METADATA_TRANSFORM_KEY,
        new_transform_key="registered",
    )

    return params


@pytest.mark.private_data
@pytest.mark.parametrize(
    "dataset",
    [ds for ds in datasets if ds["visibility"] == "private"],
)
def test_pairwise_reg_against_sample_gt(dataset):
    pass


#     get_msims_from_dataset(dataset)

# get overlap bboxes

# # assert matrix
# assert np.allclose(
#     # p["transform"].sel(t=0, x_in=[0, 1], x_out=[0, 1]),
#     p["transform"].sel(t=0, x_in=["x", "y"], x_out=["x", "y"]),
#     np.array([[1.0, 0.0], [0.0, 1.0]]),
#     atol=0.05,
# )

# gt_shift = xr.DataArray(
#     [2.5, 7.5],
#     dims=["x_in"],
#     coords={"x_in": ["y", "x"]},
# )
# tolerance = 1.5

# # somehow antspy sporadically yields different results in ~1/10 times
# if pairwise_reg_func != registration.registration_ANTsPy:
#     # assert offset
#     assert np.allclose(
#         p["transform"].sel(t=0, x_in=["y", "x"], x_out="1") - gt_shift,
#         np.zeros((2,)),
#         atol=tolerance,
#     )

# ##### test groupwise registration

# ### test for different dtypes and normalizations

# msimss = [msims]

# sim_extrema = [(float(sim.min()), float(sim.max())) for sim in sims]

# for out_range, dtype in zip(
#     [(-1, 1), (0, 1), (-300, 0)], [np.float32, np.float32, np.int32]
# ):
#     msimss += [
#         [
#             msi_utils.get_msim_from_sim(
#                 xr.apply_ufunc(
#                     rescale_intensity,
#                     sim,
#                     dask="allowed",
#                     kwargs={
#                         "in_range": sim_extrema[isim],
#                         "out_range": out_range,
#                     },
#                     keep_attrs=True,
#                 ).astype(dtype),
#                 scale_factors=[],
#             )
#             for isim, sim in enumerate(sims)
#         ]
#     ]

# for msims in msimss:
#     p = registration.register(
#         [msims[0], msims[1]],
#         registration_binning={dim: 1 for dim in spatial_dims},
#         transform_key=METADATA_TRANSFORM_KEY,
#         pairwise_reg_func=pairwise_reg_func,
#     )

#     # for groupwise registration, check relative position of a control point
#     ctrl_pt = np.zeros((2,))
#     ctrl_pts_t = [
#         transformation.transform_pts([ctrl_pt], affine.squeeze())[0]
#         for affine in p
#     ]
#     rel_pos = ctrl_pts_t[0] - ctrl_pts_t[1]

#     # somehow antspy sporadically yields different results in ~1/10 times
#     if pairwise_reg_func != registration.registration_ANTsPy:
#         # assert offset
#         assert np.allclose(
#             rel_pos,
#             # np.array([2.5, 7.5]),
#             gt_shift,
#             atol=1.5,
#         )
