"""
Neuroglancer viewer state for multiview-stitcher images.

This module is deliberately free of any UI, plotting, filesystem or HTTP
server dependency so that it can be imported both on CPython and inside the
browser runtime (Pyodide), where the same viewer state is generated for
locally computed data. `vis_utils` re-exports everything here, so
`vis_utils.generate_neuroglancer_json` keeps working unchanged.
"""

import json
import urllib.parse

import numpy as np

from multiview_stitcher import ngff_utils, spatial_image_utils

# Public Neuroglancer instance used when no custom `neuroglancer_url` is given.
_DEFAULT_NEUROGLANCER_URL = "https://neuroglancer-demo.appspot.com"


def _affine_to_neuroglancer_source_transform(
    affine, sdims, output_spacing
):
    """
    Convert a physical-space affine to a Neuroglancer source transform.

    OME-Zarr scale and translation already map pixel coordinates into the
    source coordinate space. Neuroglancer rescales source-transform linear
    coefficients from input to output dimension scales internally, but its
    translation coefficients are expressed directly in output coordinate units.
    """
    affine = np.array(affine, dtype=float, copy=True)
    affine_ndim = affine.shape[-1] - 1
    affine_sdims = sdims[-affine_ndim:]
    output_spacing_array = np.array(
        [output_spacing[dim] for dim in affine_sdims]
    )
    affine[:-1, -1] = affine[:-1, -1] / output_spacing_array
    return affine


def generate_neuroglancer_json(
    ome_zarr_paths: list[str],
    ome_zarr_urls: list[str],
    sims: list = None,
    transform_key: str = None,
    channel_coord: str = None,
    single_layer: bool = False,
    contrast_limits: tuple = None,
    layer_dicts: list[dict] = None,
    global_dict: dict = None,
    layout: str = None,
):
    virtual_ome_zarrs = ome_zarr_paths is None

    # read the first multiscales
    if virtual_ome_zarrs:
        if sims is None:
            raise ValueError(
                "sims must be provided when ome_zarr_paths is None."
            )
        ngff_multiscales = None
        ngff_images = None
        ome_zarr_sim0 = sims[0]
        sim = ome_zarr_sim0
    else:
        ngff_multiscales = [
            ngff_utils.read_ngff_multiscales(ome_zarr_path)
            for ome_zarr_path in ome_zarr_paths
        ]
        ngff_images = [
            multiscales.images[0]
            for multiscales in ngff_multiscales
        ]
        ome_zarr_sim0 = ngff_utils.ngff_image_to_sim(
            ngff_images[0],
            transform_key=spatial_image_utils.DEFAULT_TRANSFORM_KEY,
        )
        sim = ome_zarr_sim0
    sdims = spatial_image_utils.get_spatial_dims_from_sim(sim)
    ndim = len(sdims)
    dims = sim.dims
    spacing = spatial_image_utils.get_spacing_from_sim(sim)

    if sims is not None:
        if transform_key is None and not virtual_ome_zarrs:
            raise ValueError(
                "transform_key must be provided if sims are given"
            )

        spacings_per_sim = []
        if transform_key is None:
            full_affines = [None for _ in sims]
            for source_sim in sims:
                spacings_per_sim.append(
                    spatial_image_utils.get_spacing_from_sim(source_sim)
                )
        else:
            full_affines = [np.eye(len(dims) + 1) for _ in sims]
            for isim, registered_sim in enumerate(sims):

                if virtual_ome_zarrs:
                    sim_ome_zarr = registered_sim
                else:
                    sim_ome_zarr = (
                        ome_zarr_sim0
                        if isim == 0
                        else ngff_utils.ngff_image_to_sim(
                            ngff_images[isim],
                            transform_key=(
                                spatial_image_utils.DEFAULT_TRANSFORM_KEY
                            ),
                        )
                    )
                spacing_zarr = spatial_image_utils.get_spacing_from_sim(
                    sim_ome_zarr
                )
                spacing_isim = spacing_zarr
                spacings_per_sim.append(spacing_isim)

                affine = spatial_image_utils.get_affine_from_sim(
                    registered_sim, transform_key=transform_key
                )
                if "t" in affine.dims:
                    affine = affine.sel(t=0)

                # Compose a correction that maps from OME-Zarr physical coordinates to
                # in-memory physical coordinates before applying the registered affine.
                # This is needed when the user has modified origin/spacing of the in-memory
                # sim relative to what is stored in the OME-Zarr on disk.
                affine_np = np.array(affine, dtype=float)
                affine_ndim = affine_np.shape[-1] - 1
                affine_sdims = sdims[-affine_ndim:]
                origin_zarr = spatial_image_utils.get_origin_from_sim(sim_ome_zarr)
                origin_mem = spatial_image_utils.get_origin_from_sim(registered_sim)
                spacing_mem = spatial_image_utils.get_spacing_from_sim(registered_sim)
                correction = np.eye(affine_ndim + 1)
                for i, dim in enumerate(affine_sdims):
                    scale = spacing_mem[dim] / spacing_zarr[dim]
                    correction[i, i] = scale
                    correction[i, affine_ndim] = (
                        origin_mem[dim] - origin_zarr[dim] * scale
                    )
                affine_np = affine_np @ correction

                affine_ng = _affine_to_neuroglancer_source_transform(
                    affine_np,
                    sdims=sdims,
                    output_spacing=spacing_isim,
                )
                affine_ndim = affine_ng.shape[-1] - 1
                full_affines[isim][
                    -affine_ndim - 1 :, -affine_ndim - 1 :
                ] = affine_ng
    else:
        full_affines = [None for _ in ome_zarr_urls]
        spacings_per_sim = [spacing] * len(ome_zarr_urls)

    window = None
    if contrast_limits is not None:
        window = {
            "min": contrast_limits[0],
            "max": contrast_limits[1],
            "start": contrast_limits[0],
            "end": contrast_limits[1],
        }
    if "c" in dims:
        if channel_coord is None:
            channel_index = 0
        else:
            # this currently assumes that channel_coord
            # is present in all sims and at the same index
            channel_coord = str(channel_coord)
            if sims is not None:
                channel_coords = sims[0].coords["c"].values
            elif not virtual_ome_zarrs:
                omero = getattr(ngff_multiscales[0].metadata, "omero", None)
                channel_coords = (
                    [channel.label for channel in omero.channels]
                    if omero is not None
                    else sim.coords["c"].values
                )
            else:
                channel_coords = sim.coords["c"].values
            channel_index = [str(c) for c in channel_coords].index(channel_coord)
    else:
        channel_index = 0

    if not virtual_ome_zarrs:
        unit_specs = {
            "meter": (1, "m"),
            "millimeter": (1e-3, "m"),
            "micrometer": (1e-6, "m"),
            "nanometer": (1e-9, "m"),
            "second": (1, "s"),
            "millisecond": (1e-3, "s"),
            "microsecond": (1e-6, "s"),
            "nanosecond": (1e-9, "s"),
        }
        dimension_specs_per_source = []
        for ngff_image in ngff_images:
            dimension_specs = {}
            for dim in dims:
                unit = (ngff_image.axes_units or {}).get(dim)
                scale, ng_unit = unit_specs.get(unit, (1, unit or ""))
                dimension_specs[dim] = [
                    float(ngff_image.scale.get(dim, 1)) * scale,
                    ng_unit,
                ]
            dimension_specs_per_source.append(dimension_specs)
    else:
        dimension_specs_per_source = []
        for spacing_isim in spacings_per_sim:
            dimension_specs_per_source.append({
                dim: [
                    float(spacing_isim[dim]) if dim in sdims else 1,
                    "um" if dim in sdims else "",
                ]
                for dim in dims
            })

    output_dimensions = {
        dim: dimension_specs_per_source[0][dim]
        for dim in dims
    }

    if layout is None:
        layout = "xy" if ndim == 2 else "4panel"

    ng_config = {
        "dimensions": output_dimensions,
        "displayDimensions": sdims[::-1],
        "layerListPanel": {"visible": True},
        # 'position': [center[idim] for idim, dim in enumerate(sdims)],
        # "concurrentDownloads": 100, # leave at default
        "layout": layout,
    }

    if not single_layer:
        ng_config["layers"] = [
            {
                # "type": "image",
                "source": {
                    "url": f"{url}",
                    "transform": {
                        # neuroglancer drops last row of homogeneous matrix
                        "matrix": [
                            [float(value) for value in row]
                            for row in full_affines[iview][:-1]
                        ],
                        "outputDimensions": {
                            (dim if dim != "c" else "c'"): dimension_specs_per_source[
                                iview
                            ][dim]
                            for dim in dims
                        },
                    }
                    if full_affines[iview] is not None
                    else {},
                },
                "localDimensions": {"c'": [1, ""]} if "c" in dims else {},
                "localPosition": [channel_index] if "c" in dims else [],
                # 'localPosition': [0 for nsdim in nsdims] + [centers[iview][idim] for idim, dim in enumerate(sdims)],
                "tab": "rendering",
                "opacity": 0.6,
                # 'volumeRendering': 'on',
                "name": f"View {iview}",
            }
            | (
                {
                    "shaderControls": {
                        "normalized": {
                            "range": [window["min"], window["max"]],
                            "window": [window["start"], window["end"]],
                        },
                    },
                }
                if window is not None
                else {}
            )
            for iview, url in enumerate(ome_zarr_urls)
        ]

    else:
        ng_config["layers"] = [
            {
                # "type": "image",
                "source": [
                    {
                        "url": f"{url}",
                    }
                    | (
                        {
                            "transform": {
                                # neuroglancer drops last row of homogeneous matrix
                                "matrix": [
                                    [float(value) for value in row]
                                    for row in full_affines[iview][:-1]
                                ],
                                "outputDimensions": {
                                    (dim if dim != "c" else "c'"): dimension_specs_per_source[
                                        iview
                                    ][dim]
                                    for dim in dims
                                },
                            },
                        }
                        if full_affines[iview] is not None
                        else {}
                    )
                    for iview, url in enumerate(ome_zarr_urls)
                ],
                "localDimensions": {"c'": [1, ""]} if "c" in dims else {},
                "localPosition": [channel_index] if "c" in dims else [],
                "tab": "rendering",
                "opacity": 0.6,
                # 'volumeRendering': 'on',
                "name": "Tiles",
            }
            | (
                {
                    "shaderControls": {
                        "normalized": {
                            "range": [window["min"], window["max"]],
                            "window": [window["start"], window["end"]],
                        },
                    },
                }
                if window is not None
                else {}
            )
        ]

    # allow to overwrite / add settings for each layer
    if layer_dicts is not None:
        for il, layer_dict in enumerate(layer_dicts):
            ng_config["layers"][il] = {
                **ng_config["layers"][il],
                **layer_dict,
            }

    # allow to overwrite / add global settings
    if global_dict is not None:
        ng_config = {**ng_config, **global_dict}

    # import pprint
    # pprint.pprint(ng_config)
    return ng_config


def get_neuroglancer_url(ng_json, neuroglancer_url=None):
    """Build a Neuroglancer link that encodes `ng_json` as URL state.

    Parameters
    ----------
    ng_json : dict
        Neuroglancer viewer state, as produced by
        `generate_neuroglancer_json`.
    neuroglancer_url : str, optional
        Base URL of the Neuroglancer instance to link to, e.g. a
        self-hosted deployment. By default, the public demo instance at
        `_DEFAULT_NEUROGLANCER_URL` is used.
    """
    base_url = (neuroglancer_url or _DEFAULT_NEUROGLANCER_URL).rstrip("/")
    ng_url = base_url + "/#!" + urllib.parse.quote(
        json.dumps(ng_json, separators=(",", ":"))
    )
    return ng_url
