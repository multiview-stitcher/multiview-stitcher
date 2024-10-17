# Data formats

!!! note
    `multiview-stitcher` works with any numpy-like input arrays. Therefore, as long as the data can be read into a numpy array, it can be used with `multiview-stitcher`.

For attaching metadata to arrays, multiview-stitcher works with [SpatialImage](https://github.com/spatial-image/spatial-image) objects (with additional transform matrices attached). They can be constructed from Numpy, Dask or CuPy arrays as such:

```python
from multiview_stitcher import spatial_image_utils as si_utils
sim = si_utils.get_sim_from_array(
    tile_array,
    dims=["c", "y", "x"],
    scale={'y': 0.5, 'x': 0.5},
    translation={"y": 30, "x": 50},
    transform_key="stage_metadata",
    c_coords=['DAPI', 'GFP'],
)
```

A multiscale version of this object is represented by instances of [MultiscaleSpatialImage](https://github.com/spatial-image/multiscale-spatial-image), which can be created as such:

```python
from multiview_stitcher import msi_utils
msim = msi_utils.get_msim_from_sim(sim, scale_factors=[2, 4])
```

The following code can be used to extract a given scale from a multiscale image:

```python
sim = msi_utils.get_sim_from_msim(msim, scale="scale0")
```


## OME-Zarr

!!! note
    NGFF 0.4 (the latest OME-Zarr standard) currently only supports translation transformations. Therefore, affine transformations cannot yet be stored in OME-Zarr files.

Some support for reading and writing OME-Zarrs is provided by [multiscaleimage.MultiscaleImage](https://github.com/spatial-image/multiscale-spatial-image).

Further, `multiview_stitcher.ngff_utils` provides some convenience functions for reading and writing OME-Zarrs using [`ngff-zarr`](https://github.com/thewtex/ngff-zarr).


## Further file formats

[`bioio`](https://github.com/bioio-devs/bioio) is a very convenient library for reading a large variety of image files and it includes support for lazy loading. Here's example code of how to use `bioio` to load an image file into a tile compatible with `multiview-stitcher`:

```python
import bioio
from multiview_stitcher import spatial_image_utils as si_utils

# use bioio to load the image as a xarray.DataArray
bioio_xr = BioImage("my_file.tiff").get_xarray_dask_stack().squeeze()

sim = si_utils.get_sim_from_array(
    bioio_xr.data,
    dims=bioio_xr.dims,
    scale=si_utils.get_spatial_dims_from_sim(bioio_xr),
    translation=si_utils.get_origin_from_sim(bioio_xr),
    c_coords=bioio_xr.coords["c"].values,
    transform_key="stage_metadata",
)
```
