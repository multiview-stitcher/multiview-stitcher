# multiview-stitcher

<div align="center">

[![License BSD-3](https://img.shields.io/pypi/l/multiview-stitcher.svg?color=green)](https://github.com/multiview-stitcher/multiview-stitcher/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/multiview-stitcher.svg?color=green)](https://pypi.org/project/multiview-stitcher)
[![Python Version](https://img.shields.io/pypi/pyversions/multiview-stitcher.svg?color=green)](https://python.org)
[![tests](https://github.com/multiview-stitcher/multiview-stitcher/actions/workflows/test_and_deploy.yml/badge.svg)](https://github.com/multiview-stitcher/multiview-stitcher/actions)
[![DOI](https://zenodo.org/badge/697999800.svg)](https://zenodo.org/doi/10.5281/zenodo.13151252)

**Modern, distributed stitching toolbox for multi-view light sheet microscopy and beyond**

📖 [**Documentation**](https://multiview-stitcher.github.io/multiview-stitcher) • 🚀 [**Getting Started**](https://multiview-stitcher.github.io/multiview-stitcher/main/code_example/) • 📚 [**Examples**](https://github.com/multiview-stitcher/multiview-stitcher/tree/main/notebooks) • 🧩 [**Napari Plugin**](https://github.com/multiview-stitcher/napari-stitcher)

</div>

## Overview

`multiview-stitcher` is an **open-source modular toolbox** for distributed and tiled stitching of 2D/3D image data in Python. It provides a complete collection of algorithms to **register** and **fuse** datasets from **multi-positioning** and **multi-view** light sheet microscopy, as well as other imaging modalities like correlative cryo-EM.

### ✨ Key Features

- 🔧 **Modular framework** - Extensible architecture for custom registration and fusion algorithms
- ⚡ **Distributed processing** - Built on `dask` for scalable, chunked image processing
- 🌐 **Standards compliant** - Full integration with [NGFF specification](https://github.com/ome/ngff) and OME-Zarr
- 📊 **Rich ecosystem** - Seamless integration with `xarray`, `spatial-image`, and scientific Python tools
- 🎯 **Multi-modal support** - Light sheet microscopy, correlative cryo-EM, and more
- 👁️ **Interactive visualization** - Companion [Napari plugin](https://github.com/multiview-stitcher/napari-stitcher) for real-time visualization

### 🔬 What It Does

Transform multi-position microscopy data from overlapping tiles into seamlessly stitched, high-resolution images through:

1. **Registration** - Align overlapping image tiles with sub-pixel precision
2. **Global optimization** - Resolve registration parameters across entire datasets  
3. **Fusion** - Combine registered tiles with intelligent blending and deconvolution
4. **Visualization** - Interactive exploration of results in 2D/3D

## 🚀 Quick Start

### Installation

Install via pip:
```bash
pip install multiview-stitcher
```

Or from source:
```bash
pip install git+https://github.com/multiview-stitcher/multiview-stitcher.git
```

### Basic Usage

A complete stitching workflow in just a few lines:

```python
from multiview_stitcher import registration, fusion

# 1. Register overlapping tiles
params = registration.register(
    tiles, 
    reg_channel="DAPI",
    transform_key="stage_metadata"
)

# 2. Fuse into final image  
result = fusion.fuse(tiles, transform_key="registered")
```

<details>
<summary><b>📋 View Complete Example</b></summary>

This example walks through a complete stitching workflow:

**1. Prepare your data**
```python
import numpy as np
from multiview_stitcher import msi_utils, spatial_image_utils as si_utils

# Your image tiles (numpy, dask, cupy arrays all supported)
tile_arrays = [np.random.randint(0, 100, (2, 10, 100, 100)) for _ in range(3)]

# Define tile positions and spacing
tile_translations = [
    {"z": 2.5, "y": -10, "x": 30},
    {"z": 2.5, "y": 30, "x": 10}, 
    {"z": 2.5, "y": 30, "x": 50},
]
spacing = {"z": 2, "y": 0.5, "x": 0.5}
channels = ["DAPI", "GFP"]

# Convert to spatial images
msims = []
for tile_array, tile_translation in zip(tile_arrays, tile_translations):
    sim = si_utils.get_sim_from_array(
        tile_array, dims=["c", "z", "y", "x"],
        scale=spacing, translation=tile_translation,
        transform_key="stage_metadata", c_coords=channels
    )
    msims.append(msi_utils.get_msim_from_sim(sim, scale_factors=[]))
```

![Visualization of input tile configuration](docs/images/tile_configuration.png)

**2. Register the tiles**
```python
from dask.diagnostics import ProgressBar
from multiview_stitcher import registration

with ProgressBar():
    params = registration.register(
        msims, reg_channel="DAPI",
        transform_key="stage_metadata",
        new_transform_key="translation_registered"
    )
```

**3. Fuse into final result**
```python
from multiview_stitcher import fusion

fused_sim = fusion.fuse(
    [msi_utils.get_sim_from_msim(msim) for msim in msims],
    transform_key="translation_registered"
)

# Access as dask or numpy array
result_dask = fused_sim.data
result_numpy = fused_sim.data.compute()
```

</details>

> 💡 **More examples**: [Complete notebooks](https://github.com/multiview-stitcher/multiview-stitcher/tree/main/notebooks) | [Full documentation](https://multiview-stitcher.github.io/multiview-stitcher)

## 🧩 Ecosystem & Tools

### Napari Plugin
Interactive stitching and visualization with the powerful [napari-stitcher](https://github.com/multiview-stitcher/napari-stitcher) plugin:

![](https://github.com/multiview-stitcher/napari-stitcher/blob/dc6b571049c971709eb41064930be9b880d806f4/misc-data/20230929_screenshot.png)

*Image data by [Arthur Michaut](https://research.pasteur.fr/fr/member/arthur-michaut/) @ [Jérôme Gros Lab](https://research.pasteur.fr/fr/team/dynamic-regulation-of-morphogenesis/) @ Institut Pasteur.*

### Browser-Based Stitching
Run `multiview-stitcher` directly in your browser without installation:

**✨ Try it now:**
1. Open [JupyterLite](https://jupyter.org/try-jupyter/lab/) in a private browser window
2. Upload [notebooks/stitching_in_the_browser.ipynb](https://github.com/multiview-stitcher/multiview-stitcher/tree/main/notebooks/stitching_in_the_browser.ipynb)  
3. Upload your data files to a 'data' folder
4. Follow the notebook instructions

> **Note**: Browser-based stitching runs single-threaded and requires manual file uploads.

## 🏗️ Technical Foundation

Built on a robust stack of scientific Python tools:

- **🔧 Core Processing**: [`xarray`](https://github.com/xarray) + [`spatial-image`](https://github.com/spatial-image/spatial-image) for metadata handling
- **⚡ Distributed Computing**: [`dask`](https://github.com/dask) + [`dask-image`](https://github.com/dask-image) for chunked processing  
- **📊 Standards**: Full [NGFF specification](https://github.com/ome/ngff) compliance and OME-Zarr integration
- **🔄 Extensible**: Modular framework for custom registration and fusion algorithms

> 💡 **Extensibility**: Add custom functions for registration algorithms, fusion methods, and more. See the [extension API documentation](https://multiview-stitcher.github.io/multiview-stitcher) for details.

---

## 📖 Resources & Support

### Documentation & Examples
- 📚 [**Full Documentation**](https://multiview-stitcher.github.io/multiview-stitcher)
- 🧪 [**Example Notebooks**](https://github.com/multiview-stitcher/multiview-stitcher/tree/main/notebooks)
- 🚀 [**Quick Start Guide**](https://multiview-stitcher.github.io/multiview-stitcher/main/code_example/)

### Community & Development
- 🐛 [**Report Issues**](https://github.com/multiview-stitcher/multiview-stitcher/issues)
- 🤝 [**Contributing Guide**](docs/contributing.md)  
- 📄 [**License**](LICENSE): BSD-3-Clause

### Citation
If you use `multiview-stitcher` in your research, please cite:
> **DOI**: https://doi.org/10.5281/zenodo.13151252

---

## ⚠️ Development Status

**This project is under active development.** The API may change between versions as we work toward a stable 1.0 release. We develop in the open and welcome feedback and contributions.

### Previous Work
`multiview-stitcher` improves and replaces [MVRegFUS](https://github.com/m-albert/MVRegFus), incorporating lessons learned and modern Python ecosystem standards.
