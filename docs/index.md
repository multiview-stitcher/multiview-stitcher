---
title: Overview
---

# multiview-stitcher

[`multiview-stitcher`](https://github.com/multiview-stitcher/multiview-stitcher) is an open-source modular toolbox for distributed and tiled stitching of 2-3D image data in python. It is an extensible framework including a collection of algorithms to **register** and **fuse** small and large datasets from **multi-positioning** and **multi-view** light sheet microscopy, as well as **other modalities** such as correlative cryo-EM datasets.

For visualization, the associated [`napari-stitcher`](https://github.com/multiview-stitcher/napari-stitcher) provides visualization functionality using the Napari viewer, including a standalone widget.

With a focus on interoperability and integration with existing tools and the ecosystem, the package intends to integrate as tightly as possible with the [NGFF specification](https://github.com/ome/ngff).

It leverages [`xarray`](https://github.com/xarray) in combination with [`spatial-image`](https://github.com/spatial-data) classes for image metadata handling and [`dask`](https://github.com/dask) (and [`dask-image`](https://github.com/dask-image)) for chunked and distributed image processing.

!!! note "`multiview-stitcher` as a modular framework for registration and fusion"
    While `multiview-stitcher` contains a set of built-in functions for stitching, it is also possible to extend the package with custom functions. This can be useful for adding new registration algorithms, fusion methods, or other functionality. Have a look at the extension API documentation for more information.



## Napari plugin

There's an associated napari plugin: [napari-stitcher](https://github.com/multiview-stitcher/napari-stitcher).


## Previous work

`multiview-stitcher` improves and replaces [MVRegFus](https://github.com/m-albert/MVRegFus).


!!! note

    `multiview-stitcher` is a work in progress. The API is not yet stable.
