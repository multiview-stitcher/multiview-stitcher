[![License BSD-3](https://img.shields.io/pypi/l/multiview-stitcher.svg?color=green)](https://github.com/multiview-stitcher/multiview-stitcher/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/multiview-stitcher.svg?color=green)](https://pypi.org/project/multiview-stitcher)
[![Python Version](https://img.shields.io/pypi/pyversions/multiview-stitcher.svg?color=green)](https://python.org)
[![tests](https://github.com/multiview-stitcher/multiview-stitcher/actions/workflows/test_and_deploy.yml/badge.svg)](https://github.com/multiview-stitcher/multiview-stitcher/actions)
[![DOI](https://zenodo.org/badge/697999800.svg)](https://zenodo.org/doi/10.5281/zenodo.13151252)

<!--
[![License BSD-3](https://img.shields.io/pypi/l/multiview-stitcher.svg?color=green)](https://github.com/multiview-stitcher/multiview-stitcher/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/multiview-stitcher.svg?color=green)](https://pypi.org/project/multiview-stitcher)
[![Python Version](https://img.shields.io/pypi/pyversions/multiview-stitcher.svg?color=green)](https://python.org)
[![tests](https://github.com/multiview-stitcher/multiview-stitcher/workflows/tests/badge.svg)](https://github.com/multiview-stitcher/multiview-stitcher/actions)
[![codecov](https://codecov.io/gh/multiview-stitcher/multiview-stitcher/branch/main/graph/badge.svg)](https://codecov.io/gh/multiview-stitcher/multiview-stitcher)
-->

Documentation available [here](https://multiview-stitcher.github.io/multiview-stitcher). 📚

**Contents:** [Intro](#multiview-stitcher) • [Quickstart](#quickstart) • [Napari plugin](#napari-plugin) • [Installation](#installation) • [Recent news](#recent-news) • [Browser usage](#stitching-in-the-browser) • [Limitations](#known-limitations) • [Roadmap](#roadmap--future-plans) • [Contributing](#contributing) • [Citing](#citing-multiview-stitcher) • [License](#license)

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

## Quickstart

- [Documentation](https://multiview-stitcher.github.io/multiview-stitcher) and [code example](https://multiview-stitcher.github.io/multiview-stitcher/main/code_example/)
- Check out the [example notebooks](https://github.com/multiview-stitcher/multiview-stitcher/tree/main/notebooks).

### Code example

These code snippets walk you through a small stitching workflow consisting of
1) Preparing the input image data and metadata (tile positions, spacing, channels)
2) Registering the tiles
3) Stitching / fusing the tiles

#### 1) Prepare data for stitching

<details>
  <summary>Code snippet</summary>

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

</details>

#### 2) Register the tiles

<details>
  <summary>Code snippet</summary>

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

</details>

#### 3) Stitch / fuse the tiles

<details>
  <summary>Code snippet</summary>

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

For large datasets (>50GB, potentially with benefits already at >5GB) consider streaming the fused result directly to a zarr file using the following way to call `fusion.fuse`:

```python
from multiview_stitcher import fusion, misc_utils

fused = fusion.fuse(
    sims=[msi_utils.get_sim_from_msim(msim) for msim in msims],
    transform_key="translation_registered",
    # ... further optional args for fusion.fuse
    output_zarr_url="fused_output.ome.zarr",
    zarr_options={
        "ome_zarr": True,
        # "ngff_version": "0.4",  # optional
    },
    # optionally, we can use ray for parallelization (`pip install "ray[default]"`)
    # batch_options={
    #     "batch_func": misc_utils.process_batch_using_ray,
    #     "n_batch": 4,  # number of chunk fusions to schedule / submit at a time
    #     "batch_func_kwargs": {
    #         'num_cpus': 4  # number of processes for parallel processing to use with ray
    #     },
    # },
)
```

</details>

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

## Recent news

- Oct/25 (**v0.1.37**): Support for fusing huge datasets using `fusion.fuse(..., output_zarr_url=...)`, in which the fused result is streamed to disk in batches of independently processed chunks, circumventing any dask graph induced overhead. [Tested](https://github.com/multiview-stitcher/multiview-stitcher/blob/main/notebooks/stitching_exaspim.ipynb) on >100TB datasets!
- Oct/25 (**v0.1.34**): `register(..., reg_res_level=1)` for registering directly on downsampled data
- Aug/25 (**v0.1.30**): Multi-view fusion example [notebook](https://github.com/multiview-stitcher/multiview-stitcher/blob/main/notebooks/stitching_bigstitcher_multiview.ipynb) available.
- May/25 (**v0.1.26**): Introduced option to specify the number of parallel pairwise registrations for improved performance / memory tradeoff.
- Mar/25 (**v0.1.23**): Support for neuroglancer visualization of
  - input tiles together with their input transformations
  - registered tiles together with their registration transformations
  - fused output together with the transformations of all input tiles
- Mar/25 (**v0.1.21**): Obtained completely stable numerics for n-dimensional stack intersection calculation using `scipy.spatial.HalfspaceIntersection`.

## Citing multiview-stitcher

If you find multiview-stitcher useful please cite this repository using the following DOI (all versions): https://doi.org/10.5281/zenodo.13151252.

## Stitching in the browser

`multiview-stitcher` can run without installation in your browser. Data is processed locally in the browser and not uploaded to any server.

### Try it out

- open [JupyterLite](https://jupyter.org/try-jupyter/lab/) in a private browser window
- upload this notebook into the jupyter lab window: [notebooks/stitching_in_the_browser.ipynb](https://github.com/multiview-stitcher/multiview-stitcher/tree/main/notebooks/stitching_in_the_browser.ipynb)
- upload files to stitch into a 'data' folder in the jupyter lab window
- follow the notebook

Limitations: stitching will run with a single thread and while the code runs locally, your local file system is not directly accessible from within the browser environment

## Known limitations

1. The current implementation focuses on rigid transformations (translation, rotation). Non-rigid transformations are not supported at the moment.
1. In terms of data volumes, processing huge tiles is handled well. A large amount of tiles (e.g. more than hundreds) works but can be slow during registration, as the currently built-in global optimization method converges slowly for large numbers of tiles.
1. Open an issue if you encounter any problems or have suggestions for improvements 🙋

## Roadmap / Future plans

Some planned improvements for future releases:

1. Implement a hierarchical and parallelised global registration optimization for faster registration of datasets with large numbers of tiles (>100s).
1. Implement more built-in registration and fusion methods:
    1. Feature-based registration
    1. Multiview deconvolution-based fusion
1. The built-in option to subdivide tiles / views for working with piecewise affine transformations that account for local distortions observed in e.g. large FOV light sheet data.
1. Make multiview-stitcher available via conda-forge.
1. Open an issue if you have suggestions for improvements 🙋

## Work in progress

`multiview-stitcher` is being actively developed in the open and the API is subject to change.

## Previous work

`multiview-stitcher` improves and replaces [MVRegFUS](https://github.com/m-albert/MVRegFus).

## Issues

If you encounter any problems, please [file an issue](https://github.com/multiview-stitcher/multiview-stitcher/issues) along with a description of the problem. Interacting with the community and developers via issues is highly appreciated and encouraged 🙌

## Contributing

Contributions are very welcome 🙌

If you're looking for ideas, feel free to have a look at the open issues (e.g. those labeled with "help wanted" or "good first issue").

## License

Distributed under the terms of the BSD-3 license,
"multiview-stitcher" is free and open source software.
