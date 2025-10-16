---
title: Overview
---

# multiview-stitcher

`multiview-stitcher` is an open-source modular toolbox for distributed and tiled stitching of 2-3D image data in python. It is a collection of algorithms to **register** and **fuse** small and large datasets from **multi-positioning** and **multi-view** light sheet microscopy, as well as **other modalities** such as correlative cryo-EM datasets. As such, it shares considerable functionality with the Fiji plugin [BigStitcher](https://imagej.net/plugins/bigstitcher/), with the difference that it is designed for interoperability with the Python scientific ecosystem. This allows it to:

  - easily integrate into existing Python-based workflows (within Jupyter notebooks, scripts, etc.) 🐍
  - scale to very large datasets using mature Python tooling (using `dask`, `zarr-python`, `ray`) 🚀
  - make use of community-developed data representations (`xarray`, `spatial-image`, `multiscale-spatial-image`, `spatialdata`) 🤓
  - ensure compatibility with and optimal usage of modern file formats and standards, e.g. [OME-Zarr](https://ome-ngff.readthedocs.io/en/latest/)
  - swap in custom methods for registration and fusion that are readily available in the Python ecosystem (e.g. from `scikit-image`, `ANTs`, `elastix`, `SimpleITK`) 🔧

**👀 Visualization**: The associated [`napari-stitcher`](https://github.com/multiview-stitcher/napari-stitcher) provides visualization functionality using the Napari viewer, including a standalone widget for stitching vanilla napari image layers. Alternatively, web-based visualization of huge datasets  together with their associated transformations is supported using [neuroglancer](https://neuroglancer-docs.web.app/) (no additional installation required! See e.g. the exaSPIM example [notebook](https://github.com/multiview-stitcher/multiview-stitcher/blob/main/notebooks/stitching_exaspim.ipynb)).

**🛠️ Extensibility**: Next to the built-in functions for pairwise registration, fusion and view weighing, custom functions with a simple API can be provided by the user. Multiview-stitcher provides these functions with chunk-sized and pre-transformed image arrays, taking care of the overall stitching workflow and large data handling.

**🚀 Scalability**: The package is designed to handle very large datasets that do not fit into memory. It leverages `zarr`, `dask` and `ray` for efficient data handling and processing. For example, `multiview-stitcher` can fuse cloud-hosted exaSPIM datasets of >100TB each (see [example notebook](https://github.com/multiview-stitcher/multiview-stitcher/blob/main/notebooks/stitching_exaspim.ipynb)).

**🔄 Transformations**: multiview-stitcher supports both input and output tile transformations, as well as registration results to be full affine transformations. This includes simple shifts / translations, as well as rotation and scaling for advanced stitching or multi-view fusion. Non-rigid transformations are not supported at the moment.


## Napari plugin

There's an associated napari plugin: [napari-stitcher](https://github.com/multiview-stitcher/napari-stitcher).


## Known limitations

1. The current implementation focuses on rigid transformations (translation, rotation). Non-rigid transformations are not supported at the moment.
1. In terms of data volumes, processing huge tiles is handled well. A large amount of tiles (e.g. more than hundreds) works but can be slow during registration, as the currently built-in global optimization method converges slowly for large numbers of tiles.
1. Open an issue if you encounter any problems or have suggestions for improvements!

## Roadmap / Future plans

Some planned improvements for future releases:

1. Implement a hierarchical and parallelised global registration optimization for faster registration of datasets with large numbers of tiles (>100s).
1. Implement more built-in registration and fusion methods:
    1. Feature-based registration
    1. Multiview deconvolution-based fusion
1. The built-in option to subdivide tiles / views for working with piecewise affine transformations that account for local distortions observed in e.g. large FOV light sheet data.
1. Open an issue if you have suggestions for improvements!