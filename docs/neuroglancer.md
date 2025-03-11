# Neuroglancer (web-based)

!!! note "Neuroglancer"
    [Neuroglancer](https://github.com/google/neuroglancer) is a powerful web-based viewer for large image datasets. (Multiple) images can be visualized by simply including URLs to the image data in a neuroglancer link ([example](https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22t%22:%5B1%2C%22%22%5D%2C%22z%22:%5B4.0000000000000014e-7%2C%22%22%5D%2C%22y%22:%5B1.0833499999999673e-7%2C%22%22%5D%2C%22x%22:%5B1.0833499999999673e-7%2C%22%22%5D%7D%2C%22displayDimensions%22:%5B%22x%22%2C%22y%22%2C%22z%22%5D%2C%22position%22:%5B0%2C13%2C993.3992919921875%2C897.2318725585938%5D%2C%22crossSectionScale%22:5.365555971121942%2C%22projectionScale%22:2048%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22zarr://https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0101A/13457537.zarr%22%2C%22localDimensions%22:%7B%22c%27%22:%5B1%2C%22%22%5D%7D%2C%22localPosition%22:%5B0%5D%2C%22tab%22:%22source%22%2C%22opacity%22:0.6%2C%22shaderControls%22:%7B%22normalized%22:%7B%22range%22:%5B0%2C1200%5D%2C%22window%22:%5B0%2C1200%5D%7D%7D%2C%22name%22:%22View%200%22%7D%2C%7B%22type%22:%22image%22%2C%22source%22:%22zarr://https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0101A/13457227.zarr%22%2C%22localDimensions%22:%7B%22c%27%22:%5B1%2C%22%22%5D%7D%2C%22localPosition%22:%5B0%5D%2C%22tab%22:%22rendering%22%2C%22opacity%22:0.6%2C%22shaderControls%22:%7B%22normalized%22:%7B%22range%22:%5B0%2C1200%5D%2C%22window%22:%5B0%2C1200%5D%7D%7D%2C%22name%22:%22View%201%22%7D%5D%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%22View%200%22%7D%2C%22layout%22:%224panel%22%7D)), no installation required.

## Neuroglancer support in `multiview-stitcher`

Neuroglancer supports visualizing multiple OME-Zarr files and reads their corresponding scale and translation metadata. Additionally, neuroglancer supports attaching affine transformations to each image. This allows visualizing spatial images (`sims`) together with the transforms attached in a given `transform_key`.

multiview-stitcher provides the utility function `vis_utils.view_neuroglancer` that:
- creates a neuroglancer link given a list of OME-Zarr paths
- serves the OME-Zarrs over http in case they represent local files
- optionally includes transforms from a given `transform_key` in the `sims` in the neuroglancer link
- opens the neuroglancer link in the browser

## Examples

### Viewing a list of OME-Zarrs

```python
from multiview_stitcher import vis_utils

ome_zarr_paths = [
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0101A/13457537.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0101A/13457227.zarr"
]

vis_utils.view_neuroglancer(
    ome_zarr_paths=ome_zarr_paths,
)
```

### Viewing a list of spatial images with attached transforms

```python
from multiview_stitcher import ngff_utils, vis_utils

# if not already existing, write sims to OME-Zarrs
# (or "persist" sims to OME-Zarr)

ome_zarr_paths = [f"/tmp/ome_zarr_{isim}.zarr"
    for isim in range(len(sims))]

sims = [ngff_utils.write_sim_to_ome_zarr(sim, path)
    for zip(path, ome_zarr_paths)]

# view transform_key "registered" using neuroglancer
vis_utils.view_neuroglancer(
    ome_zarr_paths=ome_zarr_paths,
    sims=sims,
    transform_key="registered",
)
```

Limitation: Both OME-Zarr and neuroglancer currently don't allow assigning different transforms to different time points or channels.

### More examples

See the usage of `view_neuroglancer` in the [NGFF notebooks](https://github.com/multiview-stitcher/multiview-stitcher/tree/main/notebooks).
