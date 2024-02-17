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
- [x] zarr based intermediate file format for reading and writing, compatible with parallel dask workflows: [multiscale-spatial-data](https://github.com/spatial-image/multiscale-spatial-image)
- [-] czi input
  - [x] multi-positioning
  - [ ] light-sheet
- [x] tif input
- [x] tif writing

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
