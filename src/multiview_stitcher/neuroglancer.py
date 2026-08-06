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

# NGFF unit names in terms of the base units Neuroglancer expects, which it
# writes with SI prefixes rather than spelled out.
_NGFF_UNIT_SPECS = {
    "meter": (1, "m"),
    "millimeter": (1e-3, "m"),
    "micrometer": (1e-6, "m"),
    "nanometer": (1e-9, "m"),
    "second": (1, "s"),
    "millisecond": (1e-3, "s"),
    "microsecond": (1e-6, "s"),
    "nanosecond": (1e-9, "s"),
}


def _dimension_spec(scale, unit):
    """A Neuroglancer `[scale, unit]` dimension spec for an NGFF axis."""
    factor, ng_unit = _NGFF_UNIT_SPECS.get(unit, (1, unit or ""))
    return [float(scale) * factor, ng_unit]


def _time_dimension_spec(sim):
    """The Neuroglancer time dimension of the store a sim is served from.

    Neuroglancer scales each layer by the ratio between the time scale the
    store declares and the one named here, so a sim read from a store with a
    non-unity time scale has to report that scale rather than assume 1 - at a
    ratio other than 1 the viewer's time axis stops counting frames, and any
    position expressed in frames lands on the wrong one.
    """
    time_transform = ngff_utils.get_ngff_time_transform(sim)
    return _dimension_spec(time_transform["scale"], time_transform["unit"])


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


def _select_affine_sample(affine, channel_coord=None, time_index=0):
    """Reduce a transform to the single affine a Neuroglancer layer can carry.

    A transform may vary over channel or time - a channel alignment, or a
    manual placement applied to some timepoints only. A Neuroglancer source
    transform is one matrix, so the layer shows one sample of it: the channel
    the layer is drawn from, and the timepoint being viewed. The rest is
    reachable by asking again for a different sample.
    """
    if "c" in affine.dims:
        coords = [str(value) for value in affine.coords["c"].values]
        wanted = str(channel_coord) if channel_coord is not None else None
        index = coords.index(wanted) if wanted in coords else 0
        affine = affine.isel(c=index, drop=True)

    if "t" in affine.dims:
        index = int(time_index or 0)
        index = min(max(index, 0), affine.sizes["t"] - 1)
        affine = affine.isel(t=index, drop=True)

    return affine


def _project_source_transform(affine, dims, source_dims):
    """Remove synthetic dimensions from a Neuroglancer source transform."""
    indices = [dims.index(dim) for dim in source_dims]
    homogeneous = len(dims)
    keep = indices + [homogeneous]
    return affine[np.ix_(keep, keep)]


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
    source_dims: list = None,
    time_index: int = 0,
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

    if source_dims is None:
        if virtual_ome_zarrs:
            # These URLs conventionally expose the supplied sims themselves,
            # including any singleton dimensions they contain.
            source_dims = [tuple(source_sim.dims) for source_sim in sims]
        else:
            # Paths point at the original arrays, whose axes may omit the
            # singleton t/c dimensions added to our in-memory spatial images.
            source_dims = [tuple(image.dims) for image in ngff_images]
    else:
        source_dims = [tuple(source) for source in source_dims]

    if len(source_dims) != len(ome_zarr_urls):
        raise ValueError(
            "source_dims must contain one dimension sequence per OME-Zarr "
            f"URL, got {len(source_dims)} for {len(ome_zarr_urls)} URLs."
        )
    for index, source in enumerate(source_dims):
        if len(set(source)) != len(source):
            raise ValueError(
                f"source_dims[{index}] contains duplicate dimensions: "
                f"{source}."
            )
        unknown = [dim for dim in source if dim not in dims]
        if unknown:
            raise ValueError(
                f"source_dims[{index}] contains dimensions not present in "
                f"the spatial image: {unknown}."
            )

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
                affine = _select_affine_sample(
                    affine,
                    channel_coord=channel_coord,
                    time_index=time_index,
                )

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
        dimension_specs_per_source = []
        for ngff_image in ngff_images:
            dimension_specs_per_source.append({
                dim: _dimension_spec(
                    ngff_image.scale.get(dim, 1),
                    (ngff_image.axes_units or {}).get(dim),
                )
                for dim in dims
            })
    else:
        # The spatial calibration lives in the sim's coordinates, but the time
        # calibration is carried alongside them: `t` coordinates are frame
        # indices whichever time scale the store the sim came from declares.
        dimension_specs_per_source = []
        for source_sim, spacing_isim in zip(sims, spacings_per_sim):
            time_spec = _time_dimension_spec(source_sim)
            dimension_specs_per_source.append({
                dim: (
                    [float(spacing_isim[dim]), "um"]
                    if dim in sdims
                    else time_spec
                    if dim == "t"
                    else [1, ""]
                )
                for dim in dims
            })

    output_dimensions = {}
    for dim in dims:
        for source_index, source in enumerate(source_dims):
            if dim in source:
                output_dimensions[dim] = dimension_specs_per_source[
                    source_index
                ][dim]
                break

    def source_transform(index):
        affine = full_affines[index]
        if affine is None:
            return {}
        affine = _project_source_transform(
            affine, list(dims), source_dims[index]
        )
        return {
            # Neuroglancer drops the final homogeneous matrix row.
            "matrix": [
                [float(value) for value in row] for row in affine[:-1]
            ],
            "outputDimensions": {
                (dim if dim != "c" else "c'"): dimension_specs_per_source[
                    index
                ][dim]
                for dim in source_dims[index]
            },
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
                    "transform": source_transform(iview),
                },
                "localDimensions": (
                    {"c'": [1, ""]} if "c" in source_dims[iview] else {}
                ),
                "localPosition": (
                    [channel_index] if "c" in source_dims[iview] else []
                ),
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
                            "transform": source_transform(iview),
                        }
                        if full_affines[iview] is not None
                        else {}
                    )
                    for iview, url in enumerate(ome_zarr_urls)
                ],
                "localDimensions": (
                    {"c'": [1, ""]}
                    if any("c" in source for source in source_dims)
                    else {}
                ),
                "localPosition": (
                    [channel_index]
                    if any("c" in source for source in source_dims)
                    else []
                ),
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
