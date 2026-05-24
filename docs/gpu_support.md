# GPU support

Fusion supports GPU acceleration via [CuPy](https://cupy.dev/). Pass `use_cupy=True` to `fusion.fuse` and `multiview-stitcher` will transfer each input chunk to the GPU, run resampling, blending, and the fusion function there, and return a regular NumPy-backed output.

## Installation

Install CuPy following the [official instructions](https://docs.cupy.dev/en/stable/install.html).

## Usage

```python
from multiview_stitcher import fusion

fused_sim = fusion.fuse(
    images=sims, # or msims for multiscale
    transform_key="stage_metadata",
    use_cupy=True,
)

fused_sim.data.compute()
```

`use_cupy=True` also works with `output_zarr_url` for large datasets. In this case, setting `n_jobs` as in the example below allows limiting the number of parallel GPU batch processes, which can help manage GPU memory usage:

```python
from multiview_stitcher import fusion, misc_utils

fused_sim = fusion.fuse(
    images=msims,
    transform_key="stage_metadata",
    use_cupy=True,
    output_zarr_url="fused.ome.zarr",
    zarr_options={
        "ome_zarr": True
    },
    batch_options={
        "batch_func": misc_utils.process_batch_using_joblib,
        "n_batch": 20, # number of jobs to be scheduled at a time
        "batch_func_kwargs": {
            "n_jobs": 4 # number of jobs to be processed in parallel. allows limiting GPU batch processing to e.g. 4 parallel jobs
            },
    },
)
```

## What is dispatched to the GPU

| Operation | CPU | GPU |
|---|---|---|
| Affine transform | `scipy.ndimage.affine_transform` | `cupyx.scipy.ndimage.affine_transform` |
| Gaussian filter | `scipy.ndimage.gaussian_filter` | `cupyx.scipy.ndimage.gaussian_filter` |
