[![License {{cookiecutter.license}}](https://img.shields.io/pypi/l/multiview-stitcher.svg?color=green)](https://github.com/multiview-stitcher/multiview-stitcher/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/multiview-stitcher.svg?color=green)](https://pypi.org/project/multiview-stitcher)
[![Python Version](https://img.shields.io/pypi/pyversions/multiview-stitcher.svg?color=green)](https://python.org)
[![tests](https://github.com/multiview-stitcher/multiview-stitcher/actions/workflows/test_and_deploy.yml/badge.svg)](https://github.com/multiview-stitcher/multiview-stitcher/actions)
[![DOI](https://zenodo.org/badge/697999800.svg)](https://zenodo.org/doi/10.5281/zenodo.13151252)

Documentation available [here](https://multiview-stitcher.github.io/multiview-stitcher).

# multiview-stitcher

<!--
[![License BSD-3](https://img.shields.io/pypi/l/multiview-stitcher.svg?color=green)](https://github.com/multiview-stitcher/multiview-stitcher/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/multiview-stitcher.svg?color=green)](https://pypi.org/project/multiview-stitcher)
[![Python Version](https://img.shields.io/pypi/pyversions/multiview-stitcher.svg?color=green)](https://python.org)
[![tests](https://github.com/multiview-stitcher/multiview-stitcher/workflows/tests/badge.svg)](https://github.com/multiview-stitcher/multiview-stitcher/actions)
[![codecov](https://codecov.io/gh/multiview-stitcher/multiview-stitcher/branch/main/graph/badge.svg)](https://codecov.io/gh/multiview-stitcher/multiview-stitcher)
-->

`multiview-stitcher` is an open-source modular toolbox for distributed and tiled stitching of 2-3D image data in python. It is a collection of algorithms (under development) to **register** and **fuse** small and large datasets from **multi-positioning** and **multi-view** light sheet microscopy, as well as **other modalities** such as correlative cryo-EM datasets.

For visualization, the associated [`napari-stitcher`](https://github.com/multiview-stitcher/napari-stitcher) provides visualization functionality using the Napari viewer, including a standalone widget.

With a focus on interoperability and integration with existing tools and the ecosystem, the package intends to integrate as tightly as possible with the [NGFF specification](https://github.com/ome/ngff).

It leverages [`xarray`](https://github.com/xarray) in combination with [`spatial-image`](https://github.com/spatial-data) and [`multiscale-spatial-image`](https://github.com/spatial-image/multiscale-spatial-image) for image handling and [`dask`](https://github.com/dask) and [`dask-image`](https://github.com/dask-image) for chunked and distributed image processing.

## Quickstart

- [Documentation](https://multiview-stitcher.github.io/multiview-stitcher) and [code example](https://multiview-stitcher.github.io/multiview-stitcher/main/code_example/)
- Check out the [example notebooks](https://github.com/multiview-stitcher/multiview-stitcher/tree/main/notebooks).

### Code example

These code snippets walk you through a small stitching workflow consisting of
1) Preparing the input image data and metadata (tile positions, spacing, channels)
2) Registering the tiles
3) Stitching / fusing the tiles

#### 1) Prepare data for stitching


```python
import numpy as np
from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils

# input data (can be any numpy compatible array: numpy, dask, cupy, etc.)
tile_arrays = [np.random.randint(0, 100, (2, 10, 100, 100)) for _ in range(3)]

# indicate the tile offsets and spacing
tile_translations = [
    {"z": 2.5, "y": -10, "x": 30},
    {"z": 2.5, "y": 30, "x": 10},
    {"z": 2.5, "y": 30, "x": 50},
]
spacing = {"z": 2, "y": 0.5, "x": 0.5}

channels = ["DAPI", "GFP"]

# build input for stitching
msims = []
for tile_array, tile_translation in zip(tile_arrays, tile_translations):
    sim = si_utils.get_sim_from_array(
        tile_array,
        dims=["c", "z", "y", "x"],
        scale=spacing,
        translation=tile_translation,
        transform_key="stage_metadata",
        c_coords=channels,
    )
    msims.append(msi_utils.get_msim_from_sim(sim, scale_factors=[]))

# plot the tile configuration
# from multiview_stitcher import vis_utils
# fig, ax = vis_utils.plot_positions(msims, transform_key='stage_metadata', use_positional_colors=False)
```

![Visualization of input tile configuration](docs/images/tile_configuration.png)

#### 2) Register the tiles

```python
from dask.diagnostics import ProgressBar
from multiview_stitcher import registration

with ProgressBar():
    params = registration.register(
        msims,
        reg_channel="DAPI",  # channel to use for registration
        transform_key="stage_metadata",
        new_transform_key="translation_registered",
    )

# plot the tile configuration after registration
# vis_utils.plot_positions(msims, transform_key='translation_registered', use_positional_colors=False)
```

#### 3) Stitch / fuse the tiles
```python
from multiview_stitcher import fusion

fused_sim = fusion.fuse(
    [msi_utils.get_sim_from_msim(msim) for msim in msims],
    transform_key="translation_registered",
)

# get fused array as a dask array
fused_sim.data

# get fused array as a numpy array
fused_sim.data.compute()
```

## Napari plugin

There's an associated napari plugin: [napari-stitcher](https://github.com/multiview-stitcher/napari-stitcher).

![](https://github.com/multiview-stitcher/napari-stitcher/blob/dc6b571049c971709eb41064930be9b880d806f4/misc-data/20230929_screenshot.png)

Image data by [Arthur Michaut](https://research.pasteur.fr/fr/member/arthur-michaut/) @ [Jérôme Gros Lab](https://research.pasteur.fr/fr/team/dynamic-regulation-of-morphogenesis/) @ Institut Pasteur.

----------------------------------
## Installation

You can install `multiview-stitcher` via `pip` from PyPI:

    pip install multiview-stitcher

or from the source code in this github repository:

    pip install git+https://github.com/multiview-stitcher/multiview-stitcher.git

## Citing multiview-stitcher

If you find multiview-stitcher useful please cite this repository using the following DOI (all versions): https://doi.org/10.5281/zenodo.13151252.

## Stitching in the browser

`multiview-stitcher` can run without installation in your browser.

### Try it out

- open [JupyterLite](https://jupyter.org/try-jupyter/lab/) in a private browser window
- upload this notebook into the jupyter lab window: [notebooks/stitching_in_the_browser.ipynb](https://github.com/multiview-stitcher/multiview-stitcher/tree/main/notebooks/stitching_in_the_browser.ipynb)
- upload files to stitch into a 'data' folder in the jupyter lab window
- follow the notebook

#### Limitations
- stitching will run with a single thread
- while the code runs locally, your local file system is not directly accessible from within the browser environment

## Work in progress

WARNING: THIS IS WORK IN PROGRESS. `multiview-stitcher` is being developed in the open and has not reached a stable release yet. The API is subject to change.

## Previous work

`multiview-stitcher` improves and replaces [MVRegFUS](https://github.com/m-albert/MVRegFus).

## Issues

If you encounter any problems, please [file an issue](https://github.com/multiview-stitcher/multiview-stitcher/issues) along with a detailed description.

## Contributing

Contributions are welcome.

## License

Distributed under the terms of the BSD-3 license,
"multiview-stitcher" is free and open source software.
