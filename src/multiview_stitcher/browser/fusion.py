"""
Block-wise fusion that can be spread over several browser workers.

The Zarr fusion path in :mod:`multiview_stitcher.fusion` is already
embarrassingly parallel: :func:`~multiview_stitcher.fusion.prepare_block_fusion`
turns a fusion into "create the output array, then fuse block ``i``" and every
block is independent. In the browser one worker creates the output array and
the others attach to it (``create_output=False``) and fuse a disjoint subset of
blocks, so no fused pixel ever passes through JavaScript.

Every participant derives the output geometry from the *same* inputs and
options, so the block grids agree by construction.
"""

import numpy as np

from multiview_stitcher import fusion as core_fusion
from multiview_stitcher import msi_utils, ngff_utils
from multiview_stitcher import spatial_image_utils as si_utils


def _output_store_url(options):
    """Scale 0 of an OME-Zarr output lives below ``<root>/0``."""
    return f"{options.output_zarr_url.rstrip('/')}/0"


def select_input_sims(msims, options):
    """Pick, per view, the coarsest resolution level fine enough for the output.

    Mirrors what :func:`multiview_stitcher.fusion.fuse` does for multiscale
    inputs, so that a downsampled output does not read full-resolution chunks.
    """
    scale0_sims = [
        msi_utils.get_sim_from_msim(msim, scale="scale0") for msim in msims
    ]

    output_stack_properties = core_fusion.process_output_stack_properties(
        sims=scale0_sims,
        output_spacing=options.output_spacing,
        output_origin=None,
        output_shape=None,
        output_stack_properties=None,
        output_stack_mode=options.output_stack_mode,
        transform_key=options.transform_key,
    )

    sims = [
        msi_utils.get_sim_from_msim(
            msim,
            scale=(
                "scale"
                f"{msi_utils.get_res_level_from_spacing(msim, output_stack_properties['spacing'])}"
            ),
        )
        for msim in msims
    ]

    return sims, output_stack_properties


def prepare(msims, options, create_output):
    """Build the per-block fusion function for an OME-Zarr output.

    Returns the dict from
    :func:`~multiview_stitcher.fusion.prepare_block_fusion`, extended with the
    zarr array creation kwargs so the caller can write matching NGFF metadata.
    """
    if options.output_zarr_url is None:
        raise ValueError(
            "prepare() needs FusionOptions.output_zarr_url; use a preview "
            "fusion for lazy, in-memory output."
        )

    sims, output_stack_properties = select_input_sims(msims, options)

    zarr_array_creation_kwargs = (
        ngff_utils.update_zarr_array_creation_kwargs_for_ngff_version(
            options.ngff_version, {}
        )
    )

    fuse_kwargs = {
        "images": sims,
        **options.fuse_kwargs(),
        "output_stack_properties": output_stack_properties,
    }
    # output_stack_properties fully determines the geometry; passing spacing as
    # well would be redundant and is rejected downstream.
    fuse_kwargs.pop("output_spacing", None)
    fuse_kwargs.pop("output_stack_mode", None)

    info = core_fusion.prepare_block_fusion(
        _output_store_url(options),
        fuse_kwargs=fuse_kwargs,
        zarr_array_creation_kwargs=zarr_array_creation_kwargs,
        create_output=create_output,
        verbose=False,
    )
    info["zarr_array_creation_kwargs"] = zarr_array_creation_kwargs
    info["sims"] = sims
    return info


def block_ids(nblocks):
    """All block indices of a fusion, as JSON-friendly lists."""
    return [[int(i) for i in index] for index in np.ndindex(*nblocks)]


def fuse_blocks(msims, options, ids):
    """Fuse the given blocks into an output array created by another worker."""
    info = prepare(msims, options, create_output=False)
    fuse_chunk = info["func"]
    for block_id in ids:
        fuse_chunk(tuple(int(i) for i in block_id))
    return len(ids)


def finalize(msims, options, output_stack_properties):
    """Write NGFF metadata and the downsampled pyramid of a finished fusion.

    Called once, after every block has been fused.
    """
    sims, _ = select_input_sims(msims, options)

    import dask.array as da

    fused = si_utils.get_sim_from_array(
        array=da.from_zarr(_output_store_url(options)),
        dims=list(sims[0].dims),
        transform_key=options.transform_key,
        scale=output_stack_properties["spacing"],
        translation=output_stack_properties["origin"],
        c_coords=(
            sims[0].coords["c"].values if "c" in sims[0].dims else None
        ),
        t_coords=(
            sims[0].coords["t"].values if "t" in sims[0].dims else None
        ),
    )

    zarr_array_creation_kwargs = (
        ngff_utils.update_zarr_array_creation_kwargs_for_ngff_version(
            options.ngff_version, {}
        )
    )

    ngff_utils.write_sim_to_ome_zarr(
        fused,
        output_zarr_url=options.output_zarr_url,
        overwrite=False,
        zarr_array_creation_kwargs=zarr_array_creation_kwargs,
        ngff_version=options.ngff_version,
        show_progressbar=False,
    )

    return fused


def preview(msims, options):
    """Build the lazily fused msim shown in the viewer.

    Nothing is computed here: the returned multiscale image only produces
    pixels when Neuroglancer requests a chunk.
    """
    return core_fusion.fuse(images=msims, **options.fuse_kwargs())
