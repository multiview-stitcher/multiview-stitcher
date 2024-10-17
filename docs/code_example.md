# Code example

These code snippets walk you through a small stitching workflow consisting of

  1. Preparing the input image data and metadata (tile positions, spacing, channels)
  2. Registering the tiles
  3. Stitching / fusing the tiles


#### 1) Prepare data for stitching


```python
import numpy as np
from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils

# input data (can be any numpy compatible array: numpy, dask, cupy, etc.)
tile_arrays = [np.random.randint(0, 100, (2, 10, 100, 100)) for _ in range(3)]

# indicate the tile offsets and spacing
tile_translations = [
    {"z": 2.5, "y": -10, "x": 30},
    {"z": 2.5, "y": 30, "x": 10},
    {"z": 2.5, "y": 30, "x": 50},
]
spacing = {"z": 2, "y": 0.5, "x": 0.5}

channels = ["DAPI", "GFP"]

# build input for stitching
msims = []
for tile_array, tile_translation in zip(tile_arrays, tile_translations):
    sim = si_utils.get_sim_from_array(
        tile_array,
        dims=["c", "z", "y", "x"],
        scale=spacing,
        translation=tile_translation,
        transform_key="stage_metadata",
        c_coords=channels,
    )
    msims.append(msi_utils.get_msim_from_sim(sim, scale_factors=[]))

# plot the tile configuration
# from multiview_stitcher import vis_utils
# fig, ax = vis_utils.plot_positions(msims, transform_key='stage_metadata', use_positional_colors=False)
```

![Visualization of input tile configuration](images/tile_configuration.png)

#### 2) Register the tiles

```python
from dask.diagnostics import ProgressBar
from multiview_stitcher import registration

with ProgressBar():
    params = registration.register(
        msims,
        reg_channel="DAPI",  # channel to use for registration
        transform_key="stage_metadata",
        new_transform_key="translation_registered",
    )

# plot the tile configuration after registration
# vis_utils.plot_positions(msims, transform_key='translation_registered', use_positional_colors=False)
```

#### 3) Stitch / fuse the tiles
```python
from multiview_stitcher import fusion

fused_sim = fusion.fuse(
    [msi_utils.get_sim_from_msim(msim) for msim in msims],
    transform_key="translation_registered",
)

# get fused array as a dask array
fused_sim.data

# get fused array as a numpy array
fused_sim.data.compute()
```
