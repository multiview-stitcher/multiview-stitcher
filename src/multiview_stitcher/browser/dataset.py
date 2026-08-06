"""Opening OME-Zarr inputs in the browser (and in tests) as msims."""

from multiview_stitcher import msi_utils, ngff_utils
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.browser import czi as browser_czi
from multiview_stitcher.browser import example_data
from multiview_stitcher.browser import store as browser_store
from multiview_stitcher.browser.specs import SourceSpec


def open_msim(source, fetch=None, transform_key=None):
    """Open one source as an msim.

    ``source`` may be an OME-Zarr URL served by the browser's service worker,
    an ordinary filesystem path, one tile of a mosaic CZI, or a generated
    example dataset. Image data is never materialised here: the msim wraps
    zarr arrays whose chunks are fetched on demand, or - for a CZI - a dask
    array whose subblocks are read on demand.
    """
    url = source.url if isinstance(source, SourceSpec) else str(source)
    transform_key = transform_key or si_utils.DEFAULT_TRANSFORM_KEY

    if example_data.is_example_url(url):
        name, tile_index = example_data.parse_example_url(url)
        return example_data.build_msim(name, tile_index)

    if browser_czi.is_czi_url(url):
        return browser_czi.build_msim(url)

    resolved = browser_store.resolve_zarr_source(url, fetch=fetch)
    return ngff_utils.read_msim_from_ome_zarr(
        resolved,
        transform_key=transform_key,
        array_backend="zarr",
    )


def is_directly_servable(source):
    """Can the viewer read this source without going through Python?

    OME-Zarr behind the service worker can be streamed straight to
    Neuroglancer; anything else (a generated example, or a source that only
    exists in the Python heap) has to be exposed as a virtual OME-Zarr.
    """
    url = source.url if isinstance(source, SourceSpec) else str(source)
    return browser_store.is_http_url(url)


def open_msims(sources, fetch=None, transform_key=None):
    """Open a list of sources, preserving order."""
    return [
        open_msim(source, fetch=fetch, transform_key=transform_key)
        for source in sources
    ]


def check_compatible(msims):
    """Validate that a set of views can be registered and fused together.

    Raised early (and reported in the UI) rather than deep inside the
    registration graph, where the failure mode is much harder to read.
    """
    if not msims:
        raise ValueError("No images were found.")

    ndims = {msi_utils.get_ndim(msim) for msim in msims}
    if len(ndims) > 1:
        raise ValueError(
            f"All views must have the same dimensionality, got {sorted(ndims)}."
        )

    dims = {tuple(msi_utils.get_dims(msim)) for msim in msims}
    if len(dims) > 1:
        raise ValueError(
            f"All views must have the same dimensions, got {sorted(dims)}."
        )

    return True
