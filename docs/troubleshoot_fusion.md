# Large data

## Fusing large datasets

When working with very large(r) datasets (e.g. >50GB), using `fusion.fuse` to create a dask array and then computing it (either fully into memory or writing to disk) can lead to memory issues or take a very long time. This is because the underlying dask graph for fusing large datasets can become very large and complex, leading to high overhead in task scheduling and execution.

For fusing large datasets, it is recommended to stream the fused result directly to a zarr file using the `output_zarr_url` parameter of `fusion.fuse`. This approach processes the fusion in manageable chunks, writing each chunk directly to disk, which helps to avoid building up a large dask graph in memory. See the [code example](code_example.md).

## GPU memory issues

When using `use_cupy=True` for GPU acceleration, the dask scheduler may dispatch too many tasks to the GPU in parallel, which can lead to out-of-memory errors on the GPU. To mitigate this, you can

1) Fuse to zarr directly `fuse(..., output_zarr_url=)` and limit the number of parallel tasks (see [GPU support](gpu_support.md) for an example of how to do this or
2) Limit the number of parallel dask tasks when fusing to an in-memory dask array by setting e.g.

```python
with dask.config.set({"scheduler": "single-threaded"}):
    fused_sim = fusion.fuse(
        images=sims,
        transform_key="stage_metadata",
        use_cupy=True,
    )
```