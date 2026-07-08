# Zarr-backed vs Dask-backed fusion

If our input data already lives in Zarr, there are two simple ways to load it before fusion:

- **zarr-backed**: keep the input connected directly to the Zarr store
- **dask-backed**: wrap the same data as Dask arrays first

Both approaches are supported, and both give the same fused result. The difference is mainly how the input is represented before fusion starts.

In both cases, image data is read on demand and only from the regions needed for the current fusion step. The main practical difference is that **zarr-backed** fusion avoids an extra zarr-to-dask layer, which can keep the dask graph and computation more efficient.

## Zarr-backed fusion

Typical ways to get zarr-backed inputs:

- `si_utils.get_sim_from_array(zarr.open(...))`
- `ngff_utils.read_sim_from_ome_zarr(..., array_backend="zarr")`
- `ngff_utils.read_msim_from_ome_zarr(..., array_backend="zarr")`

This is the default for the OME-Zarr helpers.

For fusion, this is often the best choice when our data is already stored in Zarr.

## Dask-backed fusion

Typical ways to get dask-backed inputs:

- `si_utils.get_sim_from_array(da.from_zarr(...))`
- `ngff_utils.read_sim_from_ome_zarr(..., array_backend="dask")`
- `ngff_utils.read_msim_from_ome_zarr(..., array_backend="dask")`

Here, `da.from_zarr(...)` is just the Zarr-specific example. More generally, `si_utils.get_sim_from_array(...)` can also be used with other NumPy or Dask arrays.

This is a good option when we already want to keep the data inside a larger Dask workflow before fusion.

## Which one should I use?

If our input data is already in Zarr and we mainly want to fuse it, start with **zarr-backed** loading.

Use **dask-backed** loading when we specifically want Dask arrays for other processing steps before fusion.

In short:

- choose **zarr-backed** for the simplest and often most efficient path from Zarr input to fusion
- choose **dask-backed** when a Dask-native workflow is the main goal