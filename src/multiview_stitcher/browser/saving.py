"""Save session images or update their OME-Zarr registration metadata."""

from copy import deepcopy
from types import SimpleNamespace

from dataclasses import asdict

import dask.array as da
from zarr.errors import GroupNotFoundError

from multiview_stitcher import _zarr_compat, ngff_utils
from multiview_stitcher.browser import store as browser_store


def prepare(
    msims,
    sources,
    transform_key,
    target,
    metadata_only=False,
    fetch=None,
    write=None,
):
    """Validate every image before any output is written."""
    if not msims:
        raise ValueError("Select at least one image to save.")
    plans = []
    for msim, source in zip(msims, sources):
        multiscales = ngff_utils.msim_to_ngff_multiscales(
            msim,
            transform_key,
            ngff_version="0.6",
            target_coordinate_system=target,
        )
        if metadata_only:
            store = browser_store.resolve_zarr_source(source, fetch=fetch)
            root = _zarr_compat.open_zarr_group(store, mode="r")
            if root.attrs.get("ome", {}).get("version") not in (
                "0.6",
                "0.6.dev4",
            ):
                raise ValueError(
                    "Updating transforms requires OME-Zarr 0.6.dev4. "
                    "Use Save images to convert this source."
                )
            # Validate against the stored grid without changing source metadata.
            copy = SimpleNamespace(attrs=deepcopy(dict(root.attrs)))
            ngff_utils._update_v06_registration(
                copy, store, msim, transform_key, target
            )
        plans.append((msim, source, multiscales))
    return plans


def save(
    plan,
    destination,
    transform_key,
    target,
    metadata_only=False,
    fetch=None,
    write=None,
):
    """Write one validated image using the package's NGFF adapters."""
    msim, source, multiscales = plan
    url = source if metadata_only else destination
    store = (
        browser_store.open_http_store(
            url, fetch=fetch, write=write, writable=True
        )
        if browser_store.is_http_url(url, fetch=fetch)
        else url
    )
    if metadata_only:
        ngff_utils.update_ome_zarr_multiscales_metadata(
            store, msim, transform_key, target_coordinate_system=target
        )
    else:
        # The browser store cannot list keys, so write known levels directly
        # without ngff-zarr's metadata consolidation (which requires listing).
        try:
            _zarr_compat.open_zarr_group(store, mode="r")
        except (GroupNotFoundError, FileNotFoundError):
            pass
        else:
            raise ValueError("Output already exists; choose another folder.")
        root = _zarr_compat.open_zarr_group(store, mode="a", zarr_format=3)
        for image, dataset in zip(
            multiscales.images, multiscales.metadata.datasets
        ):
            data = da.asarray(image.data)
            array = root.create_array(
                dataset.path,
                shape=data.shape,
                dtype=data.dtype,
                chunks=data.chunksize,
                dimension_names=list(image.dims),
            )
            da.store(data, array, lock=False, scheduler="synchronous")
        metadata = ngff_utils._drop_none_values(asdict(multiscales.metadata))
        for key in ("extra", "omero"):
            metadata.pop(key, None)
        root.attrs["ome"] = {"version": "0.6.dev4", "multiscales": [metadata]}
