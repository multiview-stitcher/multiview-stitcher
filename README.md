# multiview-stitcher

<!--
[![License BSD-3](https://img.shields.io/pypi/l/multiview-stitcher.svg?color=green)](https://github.com/m-albert/multiview-stitcher/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/multiview-stitcher.svg?color=green)](https://pypi.org/project/multiview-stitcher)
[![Python Version](https://img.shields.io/pypi/pyversions/multiview-stitcher.svg?color=green)](https://python.org)
[![tests](https://github.com/m-albert/multiview-stitcher/workflows/tests/badge.svg)](https://github.com/m-albert/multiview-stitcher/actions)
[![codecov](https://codecov.io/gh/m-albert/multiview-stitcher/branch/main/graph/badge.svg)](https://codecov.io/gh/m-albert/multiview-stitcher)
-->

[`multiview-stitcher`](https://github.com/m-albert/multiview-sticher) is an open-source modular toolbox for distributed and tiled stitching of 2-3D image data in python. It is a collection of algorithms (under development) to **register** and **fuse** small and large datasets from **multi-positioning** and **multi-view** light sheet microscopy, as well as **other modalities** such as correlative cryo-EM datasets.

For visualization, the associated [`napari-stitcher`](https://github.com/m-albert/napari-sticher) provides visualization functionality using the Napari viewer, including a standalone widget.

With a focus on interoperability and integration with existing tools and the ecosystem, the package intends to integrate as tightly as possible with the [NGFF specification](https://github.com/ome/ngff).

It leverages [`xarray`](https://github.com/xarray) in combination with [`spatial-image`](https://github.com/spatial-data) classes for image metadata handling and [`dask`](https://github.com/dask) and [`dask-image`](https://github.com/dask-image) for chunked and distributed image processing.


### Napari plugin

There's an associated napari plugin: [napari-stitcher](https://github.com/m-albert/napari-stitcher).

### Work in progress

WARNING: THIS IS WORK IN PROGRESS. `multiview-stitcher` is being developed in the open but has not been released yet.

### Previous work

`multiview-stitcher` improves and replaces [MVRegFUS](https://github.com/m-albert/MVRegFus).


----------------------------------
## Installation

[TODO] You can install `napari-stitcher` via pip:

    `pip install https://github.com/m-albert/multiview-stitcher`

To install latest development version :

    pip install git+https://github.com/m-albert/multiview-stitcher.git


## Issues

If you encounter any problems, please [file an issue](https://github.com/m-albert/multiview-stitcher/issues) along with a detailed description.

## Contributing

Contributions are welcome! At the same time, we're still improving.

## License

Distributed under the terms of the BSD-3 license,
"multiview-stitcher" is free and open source software.
