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

`multiview_stitcher.ngff_utils` reads OME-Zarr 0.4, 0.5 and 0.6 and
writes these formats without loading an entire image into memory. The default
output remains 0.4. Version 0.4 uses Zarr v2; 0.5 and 0.6 use Zarr v3.

### Registration in OME-Zarr 0.6

Version 0.6 support targets the `ngff-zarr` 0.43 metadata model. The reader
accepts `0.6` and the `0.6.dev4` label emitted by that library; the built-in
writer also emits `0.6.dev4` (the exact schema revision), selected through
`ngff_version="0.6"`. Other prerelease labels, including `0.6rc0`, are rejected
rather than assumed compatible. Keep `ngff-zarr>=0.43,<0.44`: 0.44 changes the
storage backend and does not accept the Zarr stores used by the browser.

Pixel spacing and origin map each resolution into an `intrinsic` physical
coordinate system. A separate static affine maps those physical coordinates
into `registered` (or another chosen name). Registration is never baked into
pixels or folded into the per-level calibration.

```python
from multiview_stitcher import ngff_utils

# sim already carries a registration under this transform key.
ngff_utils.write_sim_to_ome_zarr(
    sim,
    "tile.ome.zarr",
    ngff_version="0.6",
    transform_key="registered",
    target_coordinate_system="registered",
)

msim = ngff_utils.read_msim_from_ome_zarr(
    "tile.ome.zarr",
    transform_key="registered",
    target_coordinate_system="registered",
)

# After refining msim's registration, update metadata without rewriting pixels.
ngff_utils.update_ome_zarr_multiscales_metadata(
    "tile.ome.zarr", msim, transform_key="registered",
    target_coordinate_system="registered",
)
```

Omitting `transform_key` when writing exports calibration only. On reading,
omitting `target_coordinate_system` selects the sole registration; multiple
registrations require explicit selection. Select the intrinsic system by name
to load the unregistered image. Both `array_backend="zarr"` (default) and
`array_backend="dask"` keep reads lazy.

Supported registrations are inline identity, scale, translation, rotation,
affine, and sequences of these, acting on `yx` or `zyx`. Time and channel axes
must remain unchanged by registration. Intrinsic and target axis names, order,
and units must match. Unsupported nonlinear, array-backed, indirect,
axis-mixing, or time/channel-varying registrations raise errors. Parent `scene`
collections are not traversed; pass an individual image group.

Use `msim_to_ngff_multiscales(msim, transform_key="registered",
ngff_version="0.6")` to export an existing pyramid through `ngff-zarr`.
For fused output, pass `zarr_options={"ome_zarr": True, "ngff_version": "0.6"}`
to `fusion.fuse`. Fusion resamples onto its output grid, so its output records
that grid's calibration without reapplying the input registration.

OME-Zarr 0.4/0.5 cannot store a general registration affine. The legacy
scale/translation converter rejects rotations, shear, and varying transforms;
use 0.6 to retain a full static affine.


## Further file formats

[`bioio`](https://github.com/bioio-devs/bioio) is a very convenient library for reading a large variety of image files and it includes support for lazy loading. Here's example code of how to use `bioio` to load an image file into a tile compatible with `multiview-stitcher`:

```python
from bioio import BioImage
from multiview_stitcher import spatial_image_utils as si_utils

# use bioio to load the image as a xarray.DataArray
bioio_xr = BioImage("my_file.tiff").get_xarray_dask_stack().squeeze()

# ensure that dimension names are lowercase (expected by the get_sim_from_array function)
bioio_xr = bioio_xr.rename(
    {dim: dim.lower() for dim in bioio_xr.dims}
    )

sim = si_utils.get_sim_from_array(
    bioio_xr.data,
    dims=bioio_xr.dims,
    scale=si_utils.get_spacing_from_sim(bioio_xr),      # dict of voxel sizes for each dim
    translation=si_utils.get_origin_from_sim(bioio_xr), # dict of origin coordinates for each dim
    c_coords=bioio_xr.coords["c"].values,
    transform_key="stage_metadata",
)
```