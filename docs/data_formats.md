# Data formats

!!! note
    `multiview-stitcher` works with any numpy-like input arrays. Therefore, as long as the data can be read into a numpy array, it can be used with `multiview-stitcher`.

A SpatialImage (`sim`) in multiview-stitcher is an `xarray.DataArray` with
spatial coordinates and named affine transformations. It can wrap NumPy, Dask
or CuPy arrays:

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

A MultiscaleSpatialImage (`msim`) is an `xarray.DataTree` with resolution
levels named `scale0`, `scale1`, and so on. Create one with:

```python
from multiview_stitcher import msi_utils
msim = msi_utils.get_msim_from_sim(sim, scale_factors=[2, 4])
```

The following code can be used to extract a given scale from a multiscale image:

```python
sim = msi_utils.get_sim_from_msim(msim, scale="scale0")
```


## OME-Zarr

`multiview_stitcher.ngff_utils` provides lazy reads and chunked writes for
OME-Zarr 0.4/0.5 and a subset of the 0.6 draft. The default output is 0.4.
OME-Zarr 0.4 uses Zarr format 2; 0.5 and the supported 0.6 draft use format 3.
These are storage-format versions, distinct from the `zarr-python` version.

### Supported 0.6 draft

`ngff_version="0.6"` selects the **0.6.dev4** schema bundled with
[`ngff-zarr` 0.43](https://github.com/fideus-labs/ngff-zarr/tree/py-v0.43.0/py/ngff_zarr/spec/0.6).
The writer records `ome.version="0.6.dev4"`; the reader accepts that value and
`"0.6"` as an alias for this model. This is not support for all 0.6 revisions:
other labels, including the [0.6rc0 release candidate](https://ngff.openmicroscopy.org/specifications/dev/index.html),
are rejected. The package requires `ngff-zarr>=0.43,<0.44` to retain compatibility
with the Zarr store objects used by the browser runtime.

### Coordinate systems and registration

Each resolution level's dataset coordinate transformation maps array indices
to the **intrinsic coordinate system**, the image's native physical coordinate
system. Scale describes pixel spacing (and time sampling, when present);
translation describes the origin, including the offset needed to align pyramid
levels. The 0.6 writer represents scale followed by translation as a `sequence`.

An **additional coordinate transformation** at the multiscales level maps
intrinsic coordinates to another named coordinate system. This is where the
adapter stores a registration affine, shared by every resolution level.
The image arrays are not resampled when exporting this transformation.

The package's `transform_key` selects an affine attached to a sim or msim.
`target_coordinate_system` names a coordinate system in OME-Zarr metadata.
They need not match. The writer uses `intrinsic` and `registered` as default
coordinate-system names; NGFF does not require those literal names.

```python
from multiview_stitcher import ngff_utils

# sim has an affine stored under "affine_registered" after registration.
ngff_utils.write_sim_to_ome_zarr(
    sim, "tile.ome.zarr",
    ngff_version="0.6",
    transform_key="affine_registered",
    target_coordinate_system="registered",
)

msim = ngff_utils.read_msim_from_ome_zarr(
    "tile.ome.zarr",
    target_coordinate_system="registered",
    transform_key="affine_registered",
)

# After refining that affine, update it without rewriting image arrays.
ngff_utils.update_ome_zarr_multiscales_metadata(
    "tile.ome.zarr", msim,
    transform_key="affine_registered",
    target_coordinate_system="registered",
)
```

Omitting `transform_key` in `write_sim_to_ome_zarr` writes only dataset
calibration. On reading, omitting `target_coordinate_system` selects the sole
additional transformation; multiple candidates require explicit selection.
Selecting the intrinsic coordinate system retains calibration and stores an
identity affine under `transform_key`. Both `array_backend="zarr"` (default)
and `array_backend="dask"` keep image reads lazy.

This adapter supports inline `identity`, `scale`, `translation`, `rotation`,
`affine`, and `sequence` transformations that reduce to a static spatial affine.
Here, **static** means identical across timepoints and channels. The adapter
requires spatial axes ordered as `yx` or `zyx`, matching intrinsic/target axis
names and units, and no transformation of time or channel coordinates by the
registration affine. These are implementation limits, not general NGFF rules.
Array-backed parameters, nonlinear transformations, and paths through
intermediate coordinate systems are unsupported. Parent `scene` groups are
not traversed; pass an individual image group.

Use `msim_to_ngff_multiscales(..., ngff_version="0.6")` to convert an existing
pyramid to ngff-zarr's in-memory `Multiscales` representation. For fused output,
pass `zarr_options={"ome_zarr": True, "ngff_version": "0.6"}` to `fusion.fuse`.
Fusion resamples images onto its output grid and writes that grid's calibration.

OME-Zarr 0.4/0.5 support scale and translation coordinate transformations, but
not the general `affine` transformation type. The legacy
`sim_to_ngff_image` converter can fold a static translation into the origin;
it rejects other registration affines rather than discarding their components.


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