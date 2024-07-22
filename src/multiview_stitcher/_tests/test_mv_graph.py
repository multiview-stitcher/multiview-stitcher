import numpy as np
import pytest
from dask import compute, delayed

from multiview_stitcher import (
    io,
    msi_utils,
    mv_graph,
    sample_data,
    spatial_image_utils,
)
from multiview_stitcher.io import METADATA_TRANSFORM_KEY


@pytest.mark.parametrize(
    "ndim, overlap",
    [(ndim, overlap) for ndim in [2, 3] for overlap in [0, 1, 3]],
)
def test_overlap(ndim, overlap):
    spacing_x = 0.5
    spacing_y = 0.5
    spacing_z = 2
    sims = sample_data.generate_tiled_dataset(
        ndim=ndim,
        overlap=overlap,
        N_c=2,
        N_t=1,
        tile_size=15,
        tiles_x=3,
        tiles_y=2,
        tiles_z=2,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        spacing_z=spacing_z,
    )

    sims = [sim.sel(t=0) for sim in sims]

    stack_propss = [
        spatial_image_utils.get_stack_properties_from_sim(
            sim, transform_key=METADATA_TRANSFORM_KEY
        )
        for sim in sims
    ]

    overlap_results = []
    for isim1, _sim1 in enumerate(sims):
        for isim2, _sim2 in enumerate(sims):
            overlap_results.append(
                delayed(mv_graph.get_overlap_between_pair_of_stack_props)(
                    stack_propss[isim1],
                    stack_propss[isim2],
                )
            )

    overlap_areas = [
        overlap_result[0]
        for overlap_result in compute(overlap_results, scheduler="processes")[
            0
        ]
    ]

    overlap_areas = np.array(overlap_areas).reshape((len(sims), len(sims)))

    unique_overlap_areas = np.unique(overlap_areas)

    # remove duplicate values (because of float comparison)
    unique_overlap_areas_filtered = list(unique_overlap_areas.copy())

    for uoa in unique_overlap_areas:
        if (
            len(
                np.where(
                    np.abs(uoa - np.array(unique_overlap_areas_filtered))
                    < spacing_x / 10.0
                )[0]
            )
            > 1
        ):
            unique_overlap_areas_filtered.remove(uoa)

    unique_overlap_areas_filtered = np.array(unique_overlap_areas_filtered)

    if overlap == 0:
        assert len(unique_overlap_areas) == 1 + 1
        assert overlap_areas[0][1] == -1

    else:
        if ndim == 2:
            if overlap == 1:
                assert len(unique_overlap_areas_filtered) == 3
            else:
                assert len(unique_overlap_areas_filtered) == 4
        elif ndim == 3:
            if overlap == 1:
                assert len(unique_overlap_areas_filtered) == 3
            else:
                assert len(unique_overlap_areas_filtered) == 5

        assert np.min(overlap_areas) == -1
        assert np.max(overlap_areas) > 0
        if overlap == 1:
            assert 0 in overlap_areas

    return


def test_mv_graph_creation():
    sims = io.read_mosaic_image_into_list_of_spatial_xarrays(
        sample_data.get_mosaic_sample_data_path()
    )

    msims = [
        msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims
    ]

    view_adj_graph = mv_graph.build_view_adjacency_graph_from_msims(
        msims, transform_key=METADATA_TRANSFORM_KEY
    )

    assert len(view_adj_graph.nodes) == len(sims)
    assert len(view_adj_graph.edges) == 1

    return


def test_get_intersection_polygon_from_pair_of_stack_props():
    sims = io.read_mosaic_image_into_list_of_spatial_xarrays(
        sample_data.get_mosaic_sample_data_path()
    )

    intersection_polygon = (
        mv_graph.get_intersection_poly_from_pair_of_stack_props(
            spatial_image_utils.get_stack_properties_from_sim(
                sims[0], transform_key=METADATA_TRANSFORM_KEY
            ),
            spatial_image_utils.get_stack_properties_from_sim(
                sims[1], transform_key=METADATA_TRANSFORM_KEY
            ),
        )
    )

    assert intersection_polygon.area() > 0
