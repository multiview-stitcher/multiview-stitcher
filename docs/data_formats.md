# Data formats

For use with `multiview-stitcher`, data needs to be transformed into a [modified `multiscaleimage.MultiscaleImage`](./objects/#image).

## TIFF

Manual reading of tif files is supported with a convenience function.

## CZI

Mosaic czi file support thanks to [bioio].

## LIF

Mosaic lif file support thanks to [bioio].

## ND2

Mosaic nd2 file support thanks to [bioio].

## NGFF

Support provided by `multiscaleimage.MultiscaleImage`.

The latest NGFF spec does not fully support the transformations required for stitching applications (yet). Therefore, `multiview-stitcher` currently defines its own data format, see [here](features/Coordinate systems).
