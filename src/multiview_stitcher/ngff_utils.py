import ngff_zarr
import numpy as np
import spatial_image as si
from datatree import DataTree

from multiview_stitcher import msi_utils, param_utils
from multiview_stitcher import spatial_image_utils as si_utils


def sim_to_ngff_image(sim, transform_key):
    """
    Convert a spatial_image (multiview-stitcher flavor) into a
    ngff_image in-memory representation compatible with NGFF v0.4.

    The translational component of the affine transform associated to
    the given transform_key will be added to the
    `translate` coordinateTransformation of the NGFF image.
    """

    sdims = si_utils.get_spatial_dims_from_sim(sim)
    nsdims = si_utils.get_nonspatial_dims_from_sim(sim)

    origin = si_utils.get_origin_from_sim(sim)
    if transform_key is not None:
        transform = si_utils.get_affine_from_sim(sim, transform_key)
        for nsdim in nsdims:
            if nsdim in transform.dims:
                transform = transform.sel(
                    {
                        nsdim: transform.coords[nsdim][0]
                        for nsdim in transform.dims
                    }
                )
        transform = np.array(transform)
        transform_translation = param_utils.translation_from_affine(transform)
        for isdim, sdim in enumerate(sdims):
            origin[sdim] = origin[sdim] + transform_translation[isdim]

    ngff_im = ngff_zarr.to_ngff_image(
        sim.data,
        dims=sim.dims,
        scale=si_utils.get_spacing_from_sim(sim),
        translation=origin,
    )

    return ngff_im


def msim_to_ngff_multiscales(msim, transform_key):
    """
    Convert a multiscale_spatial_image (multiview-stitcher flavor) into a
    ngff_image in-memory representation compatible with NGFF v0.4.

    The translational component of the affine transform associated to
    the given transform_key will be added to the
    `translate` coordinateTransformation of the NGFF image(s).
    """

    ngff_ims = []
    for scale_key in msi_utils.get_sorted_scale_keys(msim):
        sim = msi_utils.get_sim_from_msim(msim, scale=scale_key)
        ngff_ims.append(sim_to_ngff_image(sim, transform_key=transform_key))

    # workaround for creating multiscale metadata
    # does this create significant overhead?
    ngff_multiscales_scales = [
        ngff_zarr.to_multiscales(ngff_im, scale_factors=[])
        for ngff_im in ngff_ims
    ]

    ngff_multiscales = ngff_zarr.Multiscales(
        ngff_ims,
        metadata=ngff_zarr.zarr_metadata.Metadata(
            axes=ngff_multiscales_scales[0].metadata.axes,
            datasets=[
                ngff_zarr.zarr_metadata.Dataset(
                    path="scale%s/image" % iscale,
                    coordinateTransformations=ngff_multiscales_scale.metadata.datasets[
                        0
                    ].coordinateTransformations,
                )
                for iscale, ngff_multiscales_scale in enumerate(
                    ngff_multiscales_scales
                )
            ],
        ),
    )

    return ngff_multiscales


def ngff_image_to_sim(ngff_im, transform_key):
    """
    Convert a ngff_image in-memory representation compatible with NGFF v0.4
    into a spatial_image (multiview-stitcher flavor).
    """

    sim = si.to_spatial_image(
        ngff_im.data,
        dims=ngff_im.dims,
        scale=ngff_im.scale,
        translation=ngff_im.translation,
    )

    sim = si_utils.ensure_dim(sim, "t")

    if "c" not in sim.dims:
        sim = sim.expand_dims(["c"])

    sdims = si_utils.get_spatial_dims_from_sim(sim)

    si_utils.set_sim_affine(
        sim,
        param_utils.affine_to_xaffine(np.eye(len(sdims) + 1), t_coords=[0]),
        transform_key=transform_key,
    )

    return sim


def ngff_multiscales_to_msim(ngff_multiscales, transform_key):
    """
    Convert a list of ngff_image in-memory representations compatible with NGFF v0.4
    into a multiscale_spatial_image (multiview-stitcher flavor).
    """

    msim_dict = {}
    for iscale, ngff_im in enumerate(ngff_multiscales.images):
        sim = ngff_image_to_sim(ngff_im, transform_key=transform_key)
        curr_scale_msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
        msim_dict[f"scale{iscale}"] = curr_scale_msim["scale0"]

    msim = DataTree.from_dict(d=msim_dict)

    return msim
