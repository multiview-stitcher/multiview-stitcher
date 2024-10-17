# Features

Below is a list of features which are either already implemented or are on the roadmap.

## Dimensionality
- [x] 2D
- [x] 3D

## Registration

### Pairwise registration methods
- [x] Phase correlation
- [x] ANTsPy
- [ ] Elastix (`itk-elastix`)
- [ ] Bead alignment
- [ ] Phase correlation for rotation + translation

### Global paramater resolution

- [x] Graph construction
- [x] Automatic determination of reference view
- [x] Parameter concatenation along graph connectivity paths
- [x] Global optimization of registration parameters from (potentially overdetermined) pairwise transforms

## Transformations

- [x] Chunked `dask_image.ndinterp.affine_transform`
- [ ] Cupy-based transform
- [ ] Chaining transformations instead of working with static coordinate systems

## Fusion

### General

- [x] Modular API to plug in different fusion and weight functions
- [ ] Support for fusion label maps
- [ ] Cupy-based fusion

### Supported fusion methods:

  - [x] Weighted average
  - [x] Maximum intensity projection
  - [ ] Multi-view deconvolution

### Supported weights:

  - [x] Linear blending
  - [x] Content-based

## Supported data formats

- [x] OME-Zarr
- [x] Zarr based intermediate file format for reading and writing, compatible with parallel dask workflows: [multiscale-spatial-data](https://github.com/spatial-image/multiscale-spatial-image)
- CZI input
  - [x] Multi-positioning
  - [ ] Light-sheet
- [x] TIF input
- [x] TIF writing

## Visualization

### Napari

See [napari-stitcher](https://github.com/multiview-stitcher/napari-stitcher).

- [x] 2D slice view: multiscale rendering
- 3D rendered view:
  - [x] Lowest scale
  - [ ] Chunked rendering
- [x] Colormaps optimized for highlighting differences between overlapping views

## Supported usage modes
- [x] As a library to build custom reconstruction workflows
- [x] Napari plugin
- [ ] Convenience function for processing on HPC
