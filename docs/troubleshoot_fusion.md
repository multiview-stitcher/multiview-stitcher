# Large data

When working with very large(r) datasets (e.g. >50GB), using `fusion.fuse` to create a dask array and then computing it (either fully into memory or writing to disk) can lead to memory issues or take a very long time. This is because the underlying dask graph for fusing large datasets can become very large and complex, leading to high overhead in task scheduling and execution.

For fusing large datasets, it is recommended to stream the fused result directly to a zarr file using the `output_zarr_url` parameter of `fusion.fuse`. This approach processes the fusion in manageable chunks, writing each chunk directly to disk, which helps to avoid building up a large dask graph in memory. See the [code example](code_example.md).