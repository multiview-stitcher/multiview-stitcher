# Features

Below is a list of features which are either already implemented or are on the roadmap.

## Registration

### Pairwise registration
- [x] phase correlation
- [x] ANTsPy
- [ ] elastix (`itk-elastix`) will be used for up to affine transformations
- [ ] bead alignment
- [ ] phase correlation for rotation + translation

### Group registration

- [x] registration graph construction
- [x] automatic determination of suitable reference view
- [x] parameter concatenation along graph connectivity paths
- [ ] global optimization of registration parameters from (potentially overdetermined) pairwise transforms
- [ ] drift correction / temporal alignment

## Transformations

- [x] chunked `dask_image.ndinterp.affine_transform`
- [ ] cupy-based transform
- [ ] chaining transformations instead of working with static coordinate systems

## Fusion

- [x] chunkwise
- [ ] modular API to plug in different fusion functions including:
- Supported weights:
  - [x] blending
  - [x] content-based
- Supported fusion methods:
  - [x] weighted average
  - [ ] multi-view deconvolution
- [ ] fusion of overlapping label maps
- [ ] GPU compatibility

## Data formats

Array-like data formats can be read into `multiview-stitcher` flavoured `spatial_image.SpatialImage` objects using `multiview_stitcher.spatial_image_utils.get_sim_from_array`.

In addition, `multiview-stitcher.io` supports reading the following data formats directly from file:

- [ ] multi-positioning file format support (thanks to the awesome `bioio`):
  - [x] czi
  - [x] lif
  - [x] nd2
- [ ] light-sheet file format support:
  - [ ] czi
- output file formats:
  - [x] tif (can be streamed into using `multiview_stitcher.io.save_sim_as_tif`)

## Visualization

### Napari
See [napari-stitcher](github.com/napari-stitcher).
- [x] 2D slice view: multiscale rendering
- 3D rendered view:
  - [x] lowest scale
  - [ ] chunked rendering
- [x] colormaps optimized for highlighting differences between overlapping views

## Dimensionality
- [x] 2d
- [x] 3d

## Supported usage modes
- [x] as a library to build custom reconstruction workflows
- [ ] predefined workflows/pipelines adapted to specific setups
- [(x)] napari plugin
- [ ] processing on HPC
