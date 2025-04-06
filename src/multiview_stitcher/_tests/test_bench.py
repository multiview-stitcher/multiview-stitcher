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
        "name": "fluo_tcyx_2x1_0",
        "image_paths": [
            test_bench_data_dir / "fluo_tcyx_2x1_0/fluo_tcyx_2x1_0.czi"
        ],
        "parameter_path": test_bench_data_dir
        / "fluo_tcyx_2x1_0/fluo_tcyx_2x1_0.zarr",
        "dims": ["t", "c", "y", "x"],
        "visibility": "private",
    },
    {
        "name": "fluo_tcyx_3x3_0_scene0",
        "image_paths": [
            test_bench_data_dir / "fluo_tcyx_3x3_0/fluo_tcyx_3x3_0.czi"
        ],
        "parameter_path": test_bench_data_dir
        / "fluo_tcyx_3x3_0/fluo_tcyx_3x3_0_scene0.zarr",
        "dims": ["t", "c", "y", "x"],
        "visibility": "private",
        "scene": 0,
    },
    {
        "name": "fluo_tcyx_3x3_0_scene1",
        "image_paths": [
            test_bench_data_dir / "fluo_tcyx_3x3_0/fluo_tcyx_3x3_0.czi"
        ],
        "parameter_path": test_bench_data_dir
        / "fluo_tcyx_3x3_0/fluo_tcyx_3x3_0_scene1.zarr",
        "dims": ["t", "c", "y", "x"],
        "visibility": "private",
        "scene": 1,
    },
    {
        "name": "fluo_yx_3x3_0",
        "image_paths": [
            test_bench_data_dir / f"fluo_yx_3x3_0/tile_{i:1d}.zarr"
            for i in range(9)
        ],
        "parameter_path": test_bench_data_dir
        / "fluo_yx_3x3_0/fluo_yx_3x3_0.zarr",
        "dims": ["y", "x"],
        "visibility": "private",
        "tolerance": 0.0005,  # pixel spacing is 0.000138
        "test_edge_pruning_method": "keep_axis_aligned",
        "reg_config": [
            {
                "registration_binning": {dim: 1 for dim in ["y", "x"]},
            },
            {
                "registration_binning": {dim: 1 for dim in ["y", "x"]},
                "pairwise_reg_func": registration.registration_ANTsPy,
                "pairwise_reg_func_kwargs": {
                    "transform_types": ["Affine"],
                },
                "groupwise_resolution_kwargs": {
                    "transform": "affine",
                },
            },
        ],
    },
    {
        "name": "em_yx_3x2_0",
        "modality": "EM",
        "image_paths": [
            test_bench_data_dir / f"em_yx_3x2_0/tile_{i:1d}.zarr"
            for i in range(6)
        ],
        "parameter_path": test_bench_data_dir / "em_yx_3x2_0/em_yx_3x2_0.zarr",
        "dims": ["y", "x"],
        "tolerance": 0.04,  # pixel spacing is 0.02 (um). this dataset is currently not performing well
        "visibility": "private",
    },
]


def read_params(filename):
    xparamss = xr.open_zarr(filename).compute()

    return [xparamss[str(ind)] for ind in range(len(xparamss))]


def write_params(params, filename):
    xparams = xr.Dataset(
        {str(ind): param for ind, param in enumerate(params)},
    )

    xparams.to_zarr(filename, mode="w")


def get_msims_from_dataset(dataset):
    if dataset["image_paths"][0].suffix == ".czi":
        scene = 0 if "scene" not in dataset else dataset["scene"]
        sims = io.read_mosaic_into_sims(
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

        registration.register(
            msims,
            transform_key=input_transform_key,
            new_transform_key=output_transform_key,
            **reg_run_config,
        )

        input_transform_key = output_transform_key

    params = [
        si_utils.get_affine_from_sim(
            msi_utils.get_sim_from_msim(msim),
            transform_key=output_transform_key,
        )
        for msim in msims
    ]

    # from multiview_stitcher import vis_utils
    # vis_utils.view_neuroglancer(
    #     sims=[msi_utils.get_sim_from_msim(msim) for msim in msims],
    #     ome_zarr_paths=dataset["image_paths"],
    #     channel_coord=0,
    #     transform_key=output_transform_key,
    # )

    return params


@pytest.mark.private_data
@pytest.mark.parametrize(
    "dataset",
    list(datasets),
    # list([d for d in datasets if d["name"] == "em_yx_3x2_0"]),
)
def test_reg_against_reference_params(dataset):
    """

    Compare registered parameters with the reference parameters.

    Strategy for parameter comparison:
    - for each overlapping pair (according to metadata transformation):
      - get points flanking the overlap bbox in metadata world coords
      - transform these into world coords using the ref params of the first ipairel
      - assume this coord cooincides for the pair
      - back project the vertices using ref params
      - transform to world coords of params to be tested
      - ideally, the coordinates should coincide again
      - check that distance between them is smaller than tolerance

    Parameters
    ----------
    dataset : dict
    """

    msims = get_msims_from_dataset(dataset)

    spacing = si_utils.get_spacing_from_sim(
        msi_utils.get_sim_from_msim(msims[0])
    )

    sdims = si_utils.get_spatial_dims_from_sim(
        msi_utils.get_sim_from_msim(msims[0])
    )

    if "tolerance" in dataset:
        tolerance = dataset["tolerance"]
    else:
        tolerance = max([spacing[dim] for dim in sdims])

    params_ref = read_params(dataset["parameter_path"])
    params_reg = register_dataset(msims, dataset)

    g_adj = mv_graph.build_view_adjacency_graph_from_msims(
        msims,
        transform_key=METADATA_TRANSFORM_KEY,
    )

    if "test_edge_pruning_method" in dataset:
        g_adj = registration.prune_view_adjacency_graph(
            g_adj,
            method=dataset["test_edge_pruning_method"],
        )

    pairs = list(g_adj.edges())

    # compare the registered parameters with the reference
    for _ipair, pair in enumerate(pairs):
        msim0 = msi_utils.multiscale_sel_coords(
            msims[pair[0]],
            {
                "t": msi_utils.get_sim_from_msim(msims[pair[0]])
                .coords["t"]
                .values[0]
            },
        )

        msim1 = msi_utils.multiscale_sel_coords(
            msims[pair[1]],
            {
                "t": msi_utils.get_sim_from_msim(msims[pair[1]])
                .coords["t"]
                .values[0]
            },
        )

        lower, upper = registration.get_overlap_bboxes(
            msi_utils.get_sim_from_msim(msim0),
            msi_utils.get_sim_from_msim(msim1),
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
            # transform vertices using the reference param of the first ipairel
            vs_ref_world = transformation.transform_pts(
                vs, params_ref[pair[0]].sel(t=t)
            )

            # assume this coord cooincides for the pair and
            # - back project the vertices using ref params
            # - re project using reg params
            vs_ref_world_ref_data_reg_worlds = []
            for ipairel in [0, 1]:
                vs_ref_world_ref_data = transformation.transform_pts(
                    vs_ref_world,
                    np.linalg.inv(params_ref[pair[ipairel]].sel(t=t)),
                )

                vs_ref_world_ref_data_reg_world = transformation.transform_pts(
                    vs_ref_world_ref_data, params_reg[pair[ipairel]].sel(t=t)
                )

                vs_ref_world_ref_data_reg_worlds.append(
                    vs_ref_world_ref_data_reg_world
                )

            # from matplotlib import pyplot as plt
            # plt.figure()
            # plt.scatter(*vs_ref_world.T, c='black')
            # plt.scatter(*vs_ref_world_ref_data_reg_worlds[0].T, c='r')
            # plt.scatter(*vs_ref_world_ref_data_reg_worlds[1].T, c='g')
            # plt.show()

            # check that the coordinates coincide again
            assert (
                np.max(
                    np.linalg.norm(
                        vs_ref_world_ref_data_reg_worlds[1]
                        - vs_ref_world_ref_data_reg_worlds[0],
                        axis=-1,
                    )
                )
                < tolerance
            )
