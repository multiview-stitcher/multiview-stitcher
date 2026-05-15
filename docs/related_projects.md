## Related stitching tools

`multiview-stitcher` sits in a broader ecosystem of excellent open-source stitching software.  
Rather than aiming to replace existing tools, it focuses on providing a **Python-native, modular API**
that integrates well with the scientific Python ecosystem (dask/zarr/xarray, napari, etc.).

The table below is a **high-level orientation** (features and workflows often overlap, and most tools can be combined in practice):

| Tool | Ecosystem | Typical use case / focus | 2D | 3D | Transform model (typical) | Out-of-core / huge data* | Automation |
| --- | --- | --- | :--: | :--: | --- | :--: | --- |
| BigStitcher | Fiji | GUI-driven multi-view + tiled microscopy workflows | ✅ | ✅ | rigid + affine | ✅ | ImageJ macros / batch |
| Ashlar | Python | multiplexed whole-slide 2D mosaics | ✅ | — | translation/rigid mosaics | limited* | CLI + Python |
| TeraStitcher | C++ | very large tiled 3D volumes | — | ✅ | translation/rigid | ✅ | CLI |
| multiview-stitcher | Python | modular registration + fusion integrated into Python workflows | ✅ | ✅ | rigid + affine | ✅ | Jupyter notebooks / Python API + napari / neuroglancer |

\* "Out-of-core / huge data" depends heavily on workflow, file formats, and output options.

### Rule of thumb

- GUI-first workflow with broad microscopy stitching functionality → **BigStitcher**
- Whole-slide multiplexed 2D mosaics with a simple CLI → **Ashlar**
- Very large 3D tiled volumes with a dedicated toolchain → **TeraStitcher**
- Stitching and fusion as a **Python building block** that plugs into existing analysis pipelines → **multiview-stitcher**

## Related software

- [BigStitcher](https://imagej.net/plugins/bigstitcher/)
- [ashlar](https://github.com/labsyspharm/ashlar)
- [TeraStitcher](https://abria.github.io/TeraStitcher/)
- [m2stitch](https://github.com/yfukai/m2stitch)

## Other cool spaces

- [ndpyramid](https://github.com/carbonplan/ndpyramid)
- [affinder](https://www.napari-hub.org/plugins/affinder)
- [bigstream](https://github.com/GFleishman/bigstream)
- [SpatialData](https://github.com/scverse/spatialdata)


