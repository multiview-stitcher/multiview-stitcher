# GPU support

Fusion supports GPU acceleration via [CuPy](https://cupy.dev/). When input arrays are dask arrays backed by CuPy, `multiview-stitcher` automatically dispatches methods used within the fusion process to their `cupyx.scipy.ndimage` equivalents.

## Installation

Install CuPy following the [official instructions](https://docs.cupy.dev/en/stable/install.html).

## Usage

To run fusion on the GPU, convert dask array chunks to CuPy before calling `fuse`, and convert back afterwards.

If you work with `SpatialImage` objects, map each image's chunks directly:

```python
import cupy as cp
from multiview_stitcher import fusion

# send chunks to GPU
for i in range(len(sims)):
    sims[i].data = sims[i].data.map_blocks(cp.asarray)

fused_sim = fusion.fuse(
    images=sims,
    transform_key="stage_metadata",
)

# retrieve fused chunks from GPU
fused_sim.data = fused_sim.data.map_blocks(cp.asnumpy)

fused_sim.data.compute()
```

If you work with `MultiscaleSpatialImage` objects, use `msi_utils.msim_map_blocks` to apply the chunk conversion across every scale while preserving transforms and other metadata:

```python
import cupy as cp
from multiview_stitcher import fusion, msi_utils

# send all multiscale chunks to GPU
msims = [msi_utils.msim_map_blocks(msim, cp.asarray) for msim in msims]

fused_msim = fusion.fuse(
    images=msims,
    transform_key="stage_metadata",
)

# retrieve all fused multiscale chunks from GPU
fused_msim = msi_utils.msim_map_blocks(fused_msim, cp.asnumpy)

fused_msim["scale0/image"].data.compute()
```

## What is dispatched to the GPU

| Operation | CPU | GPU |
|---|---|---|
| Affine transform | `scipy.ndimage.affine_transform` | `cupyx.scipy.ndimage.affine_transform` |
| Gaussian filter | `scipy.ndimage.gaussian_filter` | `cupyx.scipy.ndimage.gaussian_filter` |
