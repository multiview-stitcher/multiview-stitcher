"""
Block-wise fusion spread over several browser workers.

The Zarr fusion path in :mod:`multiview_stitcher.fusion` is already
embarrassingly parallel: :func:`~multiview_stitcher.fusion.prepare_block_fusion`
turns a fusion into "create the output array, then fuse block ``i``", and every
block is independent.

Writing those blocks in parallel from the browser works because **each zarr
chunk is its own file**. One directory handle, chosen once by the user, is
shared with every worker; a worker opens the single file it is writing, writes
it, and closes it. Concurrent writes to *distinct* files are safe and commit
individually, so there is no global flush step and no shared mutable state.
The only serialised parts are creating the arrays and writing the multiscales
metadata, and both happen once, on the session worker.

Every participant derives the output geometry from the *same* inputs and
options, so the block grids agree by construction.
"""

from copy import deepcopy
from dataclasses import asdict

import numpy as np

from multiview_stitcher import _zarr_compat, msi_utils, ngff_utils
from multiview_stitcher import fusion as core_fusion
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.browser import store as browser_store


def inherited_omero(msims):
    """Copy the first input's channel display metadata for fused output."""
    if not msims:
        return None
    sim = msi_utils.get_sim_from_msim(msims[0], scale="scale0")
    omero = msims[0].attrs.get("omero", sim.attrs.get("omero"))
    if omero is None:
        return None
    if hasattr(omero, "__dataclass_fields__"):
        omero = asdict(omero)
    return deepcopy(omero)


def _level_path(index):
    """OME-Zarr stores resolution level *i* under ``<root>/<i>``."""
    return str(index)


def array_target(options, level_path, fetch=None, write=None):
    """Where one resolution level's array lives.

    A writable zarr store when the output is a service-worker URL, or a plain
    path when it is an ordinary directory (CPython, tests).
    """
    root = str(options.output_zarr_url).rstrip("/")
    url = f"{root}/{level_path}"

    if browser_store.is_http_url(url, fetch=fetch):
        return browser_store.open_http_store(
            url, fetch=fetch, write=write, writable=True
        )
    return url


def plan_levels(msims, options):
    """Describe every resolution level of the fused output.

    Mirrors the multiscale branch of :func:`multiview_stitcher.fusion.fuse`, so
    that a pyramid written block by block matches one produced in a single
    call.
    """
    scale0_sims = [
        msi_utils.get_sim_from_msim(msim, scale="scale0") for msim in msims
    ]

    scale0_properties = core_fusion.process_output_stack_properties(
        sims=scale0_sims,
        output_spacing=options.output_spacing,
        output_origin=None,
        output_shape=None,
        output_stack_properties=None,
        output_stack_mode=options.output_stack_mode,
        transform_key=options.transform_key,
    )

    res_shapes, _, res_abs_factors = msi_utils.calc_resolution_levels(
        scale0_properties["shape"],
    )

    levels = []
    for index, (shape, abs_factors) in enumerate(
        zip(res_shapes, res_abs_factors)
    ):
        properties = {
            "shape": {dim: int(size) for dim, size in shape.items()},
            "spacing": {
                dim: scale0_properties["spacing"][dim] * abs_factors[dim]
                for dim in shape
            },
            # Match the centre-of-pixel origin convention that OME-Zarr output
            # uses for downsampled levels.
            "origin": {
                dim: scale0_properties["origin"][dim]
                + (abs_factors[dim] - 1)
                * scale0_properties["spacing"][dim]
                / 2
                for dim in shape
            },
        }

        # Fuse each output level from the coarsest input level that is still
        # fine enough, exactly as `fuse` does.
        sims = [
            msi_utils.get_sim_from_msim(
                msim,
                scale=(
                    "scale"
                    f"{msi_utils.get_res_level_from_spacing(msim, properties['spacing'])}"
                ),
            )
            for msim in msims
        ]

        levels.append(
            {
                "path": _level_path(index),
                "properties": properties,
                "sims": sims,
            }
        )

    return levels, scale0_properties, res_abs_factors


def prepare_level(
    msims, options, level_index, create_output, fetch=None, write=None
):
    """Build the per-block fusion function for one resolution level."""
    levels, _, _ = plan_levels(msims, options)
    if not 0 <= int(level_index) < len(levels):
        raise IndexError(
            f"Level {level_index} does not exist; the output has "
            f"{len(levels)} level(s)."
        )

    level = levels[int(level_index)]

    zarr_array_creation_kwargs = (
        ngff_utils.update_zarr_array_creation_kwargs_for_ngff_version(
            options.ngff_version, {}
        )
    )

    fuse_kwargs = {
        "images": level["sims"],
        **options.fuse_kwargs(),
        "output_stack_properties": level["properties"],
    }
    # The stack properties fully determine the geometry; the other output_*
    # arguments would be redundant and are rejected downstream.
    fuse_kwargs.pop("output_spacing", None)
    fuse_kwargs.pop("output_stack_mode", None)

    info = core_fusion.prepare_block_fusion(
        array_target(options, level["path"], fetch=fetch, write=write),
        fuse_kwargs=fuse_kwargs,
        zarr_array_creation_kwargs=zarr_array_creation_kwargs,
        create_output=create_output,
        # An HTTP-backed store cannot enumerate its contents, so it cannot
        # clear an existing array either. The page removes the output
        # directory before a fusion starts instead.
        overwrite=False,
        verbose=False,
    )
    info["path"] = level["path"]
    info["properties"] = level["properties"]
    return info


def block_ids(nblocks):
    """All block indices of one level, as JSON-friendly lists."""
    return [[int(i) for i in index] for index in np.ndindex(*nblocks)]


def create_output_arrays(msims, options, fetch=None, write=None):
    """Create every level's array, and list the blocks each one needs.

    Runs once on the session worker. Creating an array writes only its
    metadata document, so this is cheap; the pixels follow in parallel.
    """
    levels, _, _ = plan_levels(msims, options)

    plan = []
    for index, level in enumerate(levels):
        info = prepare_level(
            msims,
            options,
            index,
            create_output=True,
            fetch=fetch,
            write=write,
        )
        plan.append(
            {
                "level": index,
                "path": level["path"],
                "nblocks": [int(n) for n in info["nblocks"]],
                "block_ids": block_ids(info["nblocks"]),
            }
        )

    return plan


def fuse_blocks(msims, options, level, ids, fetch=None, write=None):
    """Fuse a disjoint subset of one level's blocks - the compute-worker side.

    Each block is a separate chunk file, so workers never contend for one.
    """
    info = prepare_level(
        msims,
        options,
        level,
        create_output=False,
        fetch=fetch,
        write=write,
    )
    fuse_chunk = info["func"]

    for block_id in ids:
        fuse_chunk(tuple(int(i) for i in block_id))

    return len(ids)


def write_multiscales_metadata(msims, options, fetch=None, write=None):
    """Write the OME-Zarr group metadata once every level has been fused."""
    levels, scale0_properties, res_abs_factors = plan_levels(msims, options)

    sim0 = msi_utils.get_sim_from_msim(msims[0], scale="scale0")
    nsdims = si_utils.get_nonspatial_dims_from_sim(sim0)

    coordtfs, axes = ngff_utils.calc_ngff_coordinate_transformations_and_axes(
        scale0_properties,
        res_abs_factors,
        nsdims=nsdims,
        # The fused output shares the time axis of the views it came from.
        time_transform=ngff_utils.get_ngff_time_transform(msims[0]),
    )

    root = str(options.output_zarr_url).rstrip("/")
    if browser_store.is_http_url(root, fetch=fetch):
        target = browser_store.open_http_store(
            root, fetch=fetch, write=write, writable=True
        )
    else:
        target = root

    group = _zarr_compat.open_zarr_group(
        target,
        mode="a",
        **ngff_utils.zarr_group_creation_kwargs_for_ngff_version(
            options.ngff_version
        ),
    )

    ngff_utils.write_multiscales_metadata(
        group,
        axes=axes,
        datasets=[
            {
                "path": level["path"],
                "coordinateTransformations": coordtfs[index],
            }
            for index, level in enumerate(levels)
        ],
        ngff_version=options.ngff_version,
    )

    omero = inherited_omero(msims)
    if omero is not None:
        group.attrs["omero"] = omero

    return {
        "levels": [level["path"] for level in levels],
        "shape": levels[0]["properties"]["shape"],
    }


def preview(msims, options):
    """Build the lazily fused msim shown in the viewer.

    Nothing is computed here: the returned multiscale image only produces
    pixels when Neuroglancer requests a chunk.
    """
    return core_fusion.fuse(images=msims, **options.fuse_kwargs())
