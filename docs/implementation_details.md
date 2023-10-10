# Implementation details

## (Image) data structures

### Affine transformations

Affine transformations associated to an image / view are represented as `xarray.DataArray`s with dimensions (t, x_in, x_out), typically of shape (N_tps, ndim+1, ndim+1). There's one transform per timepoint.

### In memory representation

####[spatial-image](https://github.com/spatial-image/spatial-image)
Subclasses `xarray.DataArray`, i.e. broadly speaking these are numpy/dask arrays with axis labels, coordinates and attributes. spatial-image is dask compatible for lazy loading and parallel processing.

#### [multiscale-spatial-image](https://github.com/spatial-image/multiscale-spatial-image)
  - `xarray.datatree` containing one `xarray.Dataset` per (hierarchical) spatial scale
  -  these are collections of `xarray.DataArray` which are called "data variables" and share coordinates.
  - each scale contains a `spatial-image`s as a data variable named 'image'
  - compatible with NGFF (github.com/ome/ngff)
  - can be (de-)serialized to zarr
  - also used by github.com/scverse/spatialdata

### Coordinate systems

spatial-image, multiscale-spatial-image, as well as NGFF (as of 0.4.1), [do not yet support](https://github.com/ome/ngff/issues/94#issuecomment-1656309977):

- affine transformations
- different transformations for different timepoints.

However, affine transformations are important for positioning views relatively to each other. Therefore, `spatial-image` and `multiscale-spatial-image` are used with modifications. Specifically, affine transformation parameters which transform the image into a coordinate system of a given name are attached to both:
- `spatial-image`: as key(name)/value pairs under a 'transform' attribute
- `multiscale-spatial-image`: to each scale as data variables, sharing the 't' coordinate with the associated image data variable. This is compatible with reading and writing `multiscale_spatial_image.to_zarr()` and `datatree.open_zarr()`.

In the code, coordinate systems are referred to as *transform_key* (TODO: find better name, e.g. *coordinate_system*).


## Registration using graphs

### Overlap graph
An *overlap graph* is computed from the input images (represented as a directed `networkx.DiGraph`) in which the
- nodes represent the views and
- edges represent geometrical overlap.

This graph can be used to conveniently color views for visualization (overlapping views should have different colors, but the total number of colors used shouldn't be too large, i.e. exceed 2-4).

### Reference view

A suitable reference view can be obtained from the overlap graph by e.g. choosing the view with maximal overlap to other views.

### Registration graph

A *registration graph* or list of registration pairs (TODO: clarify whether this should be a graph or a list of pairs) is obtained from the overlap graph by e.g. finding shortest overlap-weighted paths between the reference view and all other views.
