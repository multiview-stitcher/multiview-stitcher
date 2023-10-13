# Overview

[`multiview-stitcher`](https://github.com/multiview-stitcher/multiview-sticher) is an open-source modular toolbox for distributed and tiled stitching of 2-3D image data in python. It is a collection of algorithms (under development) to **register** and **fuse** small and large datasets from **multi-positioning** and **multi-view** light sheet microscopy, as well as **other modalities** such as correlative cryo-EM datasets.

For visualization, the associated [`napari-stitcher`](https://github.com/multiview-stitcher/napari-sticher) provides visualization functionality using the Napari viewer, including a standalone widget.

With a focus on interoperability and integration with existing tools and the ecosystem, the package intends to integrate as tightly as possible with the [NGFF specification](https://github.com/ome/ngff).

It leverages [`xarray`](https://github.com/xarray) in combination with [`spatial-image`](https://github.com/spatial-data) classes for image metadata handling and [`dask`](https://github.com/dask) (and [`dask-image`]https://(github.com/dask-image)) for chunked and distributed image processing.


### Napari plugin

There's an associated napari plugin: [napari-stitcher](https://github.com/napari-stitcher).

### Work in progress

WARNING: THIS IS WORK IN PROGRESS, THE API IS NOT YET STABLE.

### Previous work

`multiview-stitcher` improves and replaces [MVRegFus](https://github.com/multiview-stitcher/MVRegFus).
