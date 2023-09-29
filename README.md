# multiview-stitcher

<!--
[![License BSD-3](https://img.shields.io/pypi/l/multiview-stitcher.svg?color=green)](https://github.com/m-albert/multiview-stitcher/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/multiview-stitcher.svg?color=green)](https://pypi.org/project/multiview-stitcher)
[![Python Version](https://img.shields.io/pypi/pyversions/multiview-stitcher.svg?color=green)](https://python.org)
[![tests](https://github.com/m-albert/multiview-stitcher/workflows/tests/badge.svg)](https://github.com/m-albert/multiview-stitcher/actions)
[![codecov](https://codecov.io/gh/m-albert/multiview-stitcher/branch/main/graph/badge.svg)](https://codecov.io/gh/m-albert/multiview-stitcher)
-->

A toolbox for registering / fusing / stitching multi-view / multi-positioning image datasets in 2-3D.

There's also an associated napari plugin: [napari-stitcher](https://github.com/napari-stitcher).

WARNING: THIS IS WORK IN PROGRESS, THE API IS NOT YET STABLE.

Improves and replaces [MVRegFUS](https://github.com/m-albert/MVRegFus).


----------------------------------
## Installation

[TODO] You can install `napari-stitcher` via [pip]:

    `pip install https://github.com/m-albert/multiview-stitcher`

## Features

### Registration

#### Pairwise registration
- [x] phase correlation
- [ ] elastix (`itk-elastix`) will be used for higher transformations
- [ ] bead alignment
- [ ] phase correlation for rotation + translation

#### Group registration

- [x] registration graph construction
- [x] automatic determination of suitable reference view
- [x] parameter concatenation along graph connectivity paths
- [ ] global optimization of registration parameters from (potentially overdetermined) pairwise transforms
- [ ] drift correction / temporal alignment

### Transformation

- [x] chunked `dask_image.ndinterp.affine_transform`
- [ ] cupy-based transform

### Fusion

- [x] chunkwise
- [ ] modular API to plug in different fusion functions including:
- Supported weights:
  - [x] blending
  - [ ] content-based
- Supported fusion methods:
  - [x] weighted average
  - [ ] multi-view deconvolution
- [ ] fusion of overlapping label maps

### Data formats
- [x] zarr based intermediate file format for reading and writing, compatible with parallel dask workflows: [multiscale-spatial-data](https://github.com/spatial-image/multiscale-spatial-image)
- [-] czi input
  - [x] multi-positioning
  - [ ] light-sheet
- [x] tif input
- [x] tif writing

### Visualization

#### Napari
See [napari-stitcher](github.com/napari-stitcher).
- [x] 2D slice view: multiscale rendering
- 3D rendered view:
  - [x] lowest scale
  - [ ] chunked rendering
- [x] colormaps optimized for highlighting differences between overlapping views

### Dimensionality
- [x] 2d
- [x] 3d

### Supported usage modes
- [x] as a library to build custom reconstruction workflows
- [ ] predefined workflows/pipelines adapted to specific setups
- [(x)] napari plugin
- [ ] processing on HPC

## Implementation details

### (Image) data structures

#### Affine transformations

- affine transformations associated to an image / view are represented as `xarray.DataArray`s with dimensions (t, x_in, x_out), typically of shape (N_tps, ndim+1, ndim+1)
- one transform per timepoint

#### [spatial-image](https://github.com/spatial-image/spatial-image)
  - subclasses `xarray.DataArray`, i.e. broadly speaking these are numpy/dask arrays with axis labels, coordinates and attributes
  - dask compatible for lazy loading and parallel processing

#### [multiscale-spatial-image](https://github.com/spatial-image/multiscale-spatial-image)
  - `xarray.datatree` containing one `xarray.Dataset` per (hierarchical) spatial scale
  -  these are collections of `xarray.DataArray` which are called "data variables" and share coordinates.
  - each scale contains a `spatial-image`s as a data variable named 'image' 
  - compatible with NGFF (github.com/ome/ngff)
  - can be (de-)serialized to zarr
  - also used by github.com/scverse/spatialdata

#### Coordinate systems

The two image structures above, as well as NGFF (as of 0.4.1), [do not yet support](https://github.com/ome/ngff/issues/94#issuecomment-1656309977):
  1) affine transformations
  2) different transformations for different timepoints

However, affine transformations are important for positioning views relatively to each other. Therefore, `spatial-image` and `multiscale-spatial-image` are used with modifications. Specifically, affine transformation parameters which transform the image into a coordinate system of a given name are attached to both:
- `spatial-image`: as key(name)/value pairs under a 'transform' attribute
- `multiscale-spatial-image`: to each scale as data variables, sharing the 't' coordinate with the associated image data variable. This is compatible with reading and writing `multiscale_spatial_image.to_zarr()` and `datatree.open_zarr()`.

In the code, coordinate systems are referred to as *transform_key* (TODO: find better name, e.g. *coordinate_system*).


### Registration using graphs

#### Overlap graph
An *overlap graph* is computed from the input images (represented as a directed `networkx.DiGraph`) in which the
- nodes represent the views and
- edges represent geometrical overlap.

This graph can be used to conveniently color views for visualization (overlapping views should have different colors, but the total number of colors used shouldn't be too large, i.e. exceed 2-4).

#### Reference view

A suitable reference view can be obtained from the overlap graph by e.g. choosing the view with maximal overlap to other views.

#### Registration graph

A *registration graph* or list of registration pairs (TODO: clarify whether this should be a graph or a list of pairs) is obtained from the overlap graph by e.g. finding shortest overlap-weighted paths between the reference view and all other views.


## Ideas / things to check out

- https://github.com/carbonplan/ndpyramid
- https://www.napari-hub.org/plugins/affinder

## Related software

- [BigStitcher](https://imagej.net/plugins/bigstitcher/)
- [ashlar](https://github.com/labsyspharm/ashlar)
- [TeraStitcher](https://abria.github.io/TeraStitcher/)

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"multiview-stitcher" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

## Installation

You can install `multiview-stitcher` via [pip]:

    pip install multiview-stitcher



To install latest development version :

    pip install git+https://github.com/m-albert/multiview-stitcher.git


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"multiview-stitcher" is free and open source software
