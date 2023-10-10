# Data formats

The in-memory image representation of image datasets is based on the classes implemented in [`spatial-image`](https://github.com/spatial-data). Instances of [`multiscale-spatial-image`](https://github.com/multiscale-spatial-data) can be serialized to and from NGFF.

## NGFF

Problem: The latest NGFF spec does not fully support the transformations required for stitching applications (yet). Therefore, the question arises how to best help extending the NGFF spec and deal with this requirement for yet unsupported features, both technically and in terms of community interaction.

Therefore, `multiview-stitcher` currently defines its own data format, see [here](features/Coordinate systems).

## CZI

Mosaic czi files supports thanks to [aicsimageio].

## LIF

## Other file formats
