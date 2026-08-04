"""Opening OME-Zarr inputs in the browser (and in tests) as msims."""

from multiview_stitcher import msi_utils, ngff_utils
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.browser import store as browser_store
from multiview_stitcher.browser.specs import SourceSpec


def open_msim(source, fetch=None, transform_key=None):
    """Open one OME-Zarr as a lazily read, zarr-backed msim.

    ``source`` may be a URL served by the browser's service worker or an
    ordinary filesystem path. Image data is never materialised here: the msim
    wraps zarr arrays whose chunks are fetched on demand.
    """
    url = source.url if isinstance(source, SourceSpec) else str(source)
    transform_key = transform_key or si_utils.DEFAULT_TRANSFORM_KEY

    resolved = browser_store.resolve_zarr_source(url, fetch=fetch)
    return ngff_utils.read_msim_from_ome_zarr(
        resolved,
        transform_key=transform_key,
        array_backend="zarr",
    )


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
        raise ValueError("No OME-Zarr images were found.")

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
