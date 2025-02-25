from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import multiview_stitcher.spatial_image_utils as si_utils
from multiview_stitcher import (
    io,
    msi_utils,
    mv_graph,
    ngff_utils,
    registration,
    transformation,
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
        "image_paths": [test_bench_data_dir / "tcyx_2x1_0/tcyx_2x1_0.czi"],
        "parameter_path": test_bench_data_dir / "tcyx_2x1_0/tcyx_2x1_0.zarr",
        "dims": ["t", "c", "y", "x"],
        "visibility": "private",
    },
    {
        "name": "tcyx_3x3_0_scene0",
        "image_paths": [test_bench_data_dir / "tcyx_3x3_0/tcyx_3x3_0.czi"],
        "parameter_path": test_bench_data_dir
        / "tcyx_3x3_0/tcyx_3x3_0_scene0.zarr",
        "dims": ["t", "c", "y", "x"],
        "visibility": "private",
        "scene": 0,
    },
    {
        "name": "tcyx_3x3_0_scene1",
        "image_paths": [test_bench_data_dir / "tcyx_3x3_0/tcyx_3x3_0.czi"],
        "parameter_path": test_bench_data_dir
        / "tcyx_3x3_0/tcyx_3x3_0_scene1.zarr",
        "dims": ["t", "c", "y", "x"],
        "visibility": "private",
        "scene": 1,
    },
    {
        "name": "yx_3x3_0",
        "image_paths": [
            test_bench_data_dir / f"yx_3x3_0/tile{i:1d}.zarr" for i in range(9)
        ],
        "parameter_path": test_bench_data_dir / "yx_3x3_0/yx_3x3_0.zarr",
        "dims": ["t", "c", "y", "x"],
        "visibility": "private",
        "scene": 1,
        "reg_config": [
            {
                "reg_binning": {dim: 1 for dim in ["y", "x"]},
            },
            {
                "reg_binning": {dim: 1 for dim in ["y", "x"]},
                "pairwise_reg_func": registration.registration_ANTsPy,
                "pairwise_reg_func_kwargs": {
                    "transform_types": ["Affine"],
                },
            },
        ],
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
        scene = 0 if "scene" not in dataset else dataset["scene"]
        sims = io.read_mosaic_image_into_list_of_spatial_xarrays(
            dataset["image_paths"][0], scene_index=scene
        )

    elif dataset["image_paths"][0].suffix == ".zarr":
        sims = [
            ngff_utils.read_sim_from_ome_zarr(image_path)
            for image_path in dataset["image_paths"]
        ]

    msims = [
        msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims
    ]

    return msims


def register_dataset(msims, dataset):
    si_utils.get_spatial_dims_from_sim(msi_utils.get_sim_from_msim(msims[0]))

    reg_config = {
        "reg_channel_index": 0,
    }

    reg_run_configs = []
    if "reg_config" in dataset:
        for reg_run_config in dataset["reg_config"]:
            reg_run_configs.append(reg_config | reg_run_config)
    else:
        reg_run_configs.append(reg_config)

    input_transform_key = METADATA_TRANSFORM_KEY
    for irun, reg_run_config in enumerate(reg_run_configs):
        output_transform_key = "reg_" + str(irun)

        params = registration.register(
            msims,
            transform_key=input_transform_key,
            new_transform_key=output_transform_key,
            **reg_run_config,
        )

        input_transform_key = output_transform_key

    return params


@pytest.mark.private_data
@pytest.mark.parametrize(
    "dataset",
    [
        ds
        for ds in datasets
        if ds["visibility"] == "private" and "yx_3x3_0" in ds["name"]
    ],
)
def test_pairwise_reg_against_sample_gt(dataset):
    """

    Compare registered parameters with the reference parameters.

    Idea for parameter comparison:
    - for each overlapping pair (according to metadata transformation):
      - calculate overlap bbox vertices in metadata transformation, pts1, pts2
      - ref reg Tr1, : project pts1, project pts2
      - new reg Tn1: project pts1, project pts2
      - make sure Tr2(pts2) - Tr1(pts1) ~= Tn2(pts2) - Tn1(pts1)
    - make sure this is true for all overlapping pairs

    Parameters
    ----------
    dataset : dict
    """

    msims = get_msims_from_dataset(dataset)

    params_ref = read_params(dataset)
    params_reg = register_dataset(msims, dataset)
    # params_reg = params_ref# + np.random.random() / 100.

    g_adj = mv_graph.build_view_adjacency_graph_from_msims(
        msims,
        transform_key=METADATA_TRANSFORM_KEY,
    )

    pairs = list(g_adj.edges())
    sdims = si_utils.get_spatial_dims_from_sim(
        msi_utils.get_sim_from_msim(msims[0])
    )

    spacing = si_utils.get_spacing_from_sim(
        msi_utils.get_sim_from_msim(msims[0])
    )
    tolerance = max([spacing[dim] for dim in sdims])

    # compare the registered parameters with the reference
    for ipair, pair in enumerate(pairs):
        lower, upper = registration.get_overlap_bboxes(
            msi_utils.get_sim_from_msim(msims[pair[0]]),
            msi_utils.get_sim_from_msim(msims[pair[1]]),
            input_transform_key=METADATA_TRANSFORM_KEY,
            output_transform_key=None,
        )
        lower, upper = lower[0], upper[0]

        # get vertices of the overlap bbox
        vs = mv_graph.get_vertices_from_stack_props(
            {
                "origin": {dim: lower[idim] for idim, dim in enumerate(sdims)},
                "shape": {dim: 2 for dim in sdims},
                "spacing": {
                    dim: upper[idim] - lower[idim]
                    for idim, dim in enumerate(sdims)
                },
            }
        )

        for t in params_ref[0].coords["t"].values:
            param_displacements = []
            for _iparam, params in enumerate([params_reg, params_ref]):
                vertices_in_world_coords = [
                    transformation.transform_pts(vs, p.sel(t=t))
                    for p in [params[pair[0]], params[pair[1]]]
                ]
                param_displacements.append(
                    vertices_in_world_coords[1] - vertices_in_world_coords[0]
                )

            assert np.allclose(
                param_displacements[0], param_displacements[1], atol=tolerance
            ), f"dataset {dataset}, pair {ipair}, t={t}, {param_displacements[0]} != {param_displacements[1]}"
