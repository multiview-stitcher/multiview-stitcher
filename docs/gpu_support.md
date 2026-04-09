# GPU support

Fusion supports GPU acceleration via [CuPy](https://cupy.dev/). When input arrays are dask arrays backed by CuPy, `multiview-stitcher` automatically dispatches methods used within the fusion process to their `cupyx.scipy.ndimage` equivalents.

## Installation

Install CuPy following the [official instructions](https://docs.cupy.dev/en/stable/install.html).

## Usage

To run fusion on the GPU, convert dask array chunks to CuPy before calling `fuse`, and convert back afterwards:

```python
import cupy as cp
from multiview_stitcher import fusion

# send chunks to GPU
for i in range(len(sims)):
    sims[i].data = sims[i].data.map_blocks(cp.asarray)

fused_sim = fusion.fuse(
    sims,
    transform_key="stage_metadata",
)

# retrieve fused chunks from GPU
fused_sim.data = fused_sim.data.map_blocks(cp.asnumpy)

fused_sim.data.compute()
```

## What is dispatched to the GPU

| Operation | CPU | GPU |
|---|---|---|
| Affine transform | `scipy.ndimage.affine_transform` | `cupyx.scipy.ndimage.affine_transform` |
| Gaussian filter | `scipy.ndimage.gaussian_filter` | `cupyx.scipy.ndimage.gaussian_filter` |
