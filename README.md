[![License {{cookiecutter.license}}](https://img.shields.io/pypi/l/multiview-stitcher.svg?color=green)](https://github.com/multiview-stitcher/multiview-stitcher/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/multiview-stitcher.svg?color=green)](https://pypi.org/project/multiview-stitcher)
[![Python Version](https://img.shields.io/pypi/pyversions/multiview-stitcher.svg?color=green)](https://python.org)
[![tests](https://github.com/multiview-stitcher/multiview-stitcher/actions/workflows/test_and_deploy.yml/badge.svg)](https://github.com/multiview-stitcher/multiview-stitcher/actions)


# multiview-stitcher

<!--
[![License BSD-3](https://img.shields.io/pypi/l/multiview-stitcher.svg?color=green)](https://github.com/multiview-stitcher/multiview-stitcher/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/multiview-stitcher.svg?color=green)](https://pypi.org/project/multiview-stitcher)
[![Python Version](https://img.shields.io/pypi/pyversions/multiview-stitcher.svg?color=green)](https://python.org)
[![tests](https://github.com/multiview-stitcher/multiview-stitcher/workflows/tests/badge.svg)](https://github.com/multiview-stitcher/multiview-stitcher/actions)
[![codecov](https://codecov.io/gh/multiview-stitcher/multiview-stitcher/branch/main/graph/badge.svg)](https://codecov.io/gh/multiview-stitcher/multiview-stitcher)
-->

[`multiview-stitcher`](https://github.com/multiview-stitcher/multiview-sticher) is an open-source modular toolbox for distributed and tiled stitching of 2-3D image data in python. It is a collection of algorithms (under development) to **register** and **fuse** small and large datasets from **multi-positioning** and **multi-view** light sheet microscopy, as well as **other modalities** such as correlative cryo-EM datasets.

For visualization, the associated [`napari-stitcher`](https://github.com/multiview-stitcher/napari-stitcher) provides visualization functionality using the Napari viewer, including a standalone widget.

With a focus on interoperability and integration with existing tools and the ecosystem, the package intends to integrate as tightly as possible with the [NGFF specification](https://github.com/ome/ngff).

It leverages [`xarray`](https://github.com/xarray) in combination with [`spatial-image`](https://github.com/spatial-data) and [`multiscale-spatial-image`](https://github.com/spatial-image/multiscale-spatial-image) for image handling and [`dask`](https://github.com/dask) and [`dask-image`](https://github.com/dask-image) for chunked and distributed image processing.

## Quickstart

### Notebooks

Check out the [example notebooks](https://github.com/multiview-stitcher/multiview-stitcher/tree/main/notebooks).

### Napari plugin

There's an associated napari plugin: [napari-stitcher](https://github.com/multiview-stitcher/napari-stitcher).

![](https://github.com/multiview-stitcher/napari-stitcher/blob/dc6b571049c971709eb41064930be9b880d806f4/misc-data/20230929_screenshot.png)

Image data by [Arthur Michaut](https://research.pasteur.fr/fr/member/arthur-michaut/) @ [Jérôme Gros Lab](https://research.pasteur.fr/fr/team/dynamic-regulation-of-morphogenesis/) @ Institut Pasteur.

### Work in progress

WARNING: THIS IS WORK IN PROGRESS. `multiview-stitcher` is being developed in the open but has not been released yet.

### Previous work

`multiview-stitcher` improves and replaces [MVRegFUS](https://github.com/m-albert/MVRegFus).


----------------------------------
## Installation

You can install `multiview-stitcher` via `pip` from PyPI:

    pip install multiview-stitcher

or from the source code in this github repository:

    pip install git+https://github.com/multiview-stitcher/multiview-stitcher.git


## Issues

If you encounter any problems, please [file an issue](https://github.com/multiview-stitcher/multiview-stitcher/issues) along with a detailed description.

## Contributing

Contributions are welcome.

## License

Distributed under the terms of the BSD-3 license,
"multiview-stitcher" is free and open source software.
