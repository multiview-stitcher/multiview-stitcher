# Objects

## Image

_Modified_ instances of `multiscaleimage.MultiscaleImage`.

Modification: Each scale has at least one _named_ [affine transform parameter](#affine-transformation-parameters) attached to it as further data variables next to `scale<scale>/image`.

While instances of [`multiscale-spatial-image`](https://github.com/multiscale-spatial-data) can be serialized to and from NGFF, modified instances of `multiscaleimage.MultiscaleImage` as used by `multiview-stitcher` cannot (yet) be serialized to and from NGFF (see [here](https://github.com/ome/ngff/issues/94)), as the support for affine transforms is missing.

Example string representation:

`print(msims[0])`
```
DataTree('None', parent=None)
│   Dimensions:  ()
│   Data variables:
│       *empty*
│   Attributes:
│       multiscaleSpatialImageVersion:  1
│       multiscales:                    [{'@type': 'ngff:Image', 'axes': [{'name'...
├── DataTree('scale0')
│       Dimensions:          (t: 1, x_in: 4, x_out: 4, c: 1, z: 179, y: 1040, x: 1392)
│       Coordinates:
│         * c                (c) int64 0
│         * t                (t) int64 0
│         * x                (x) float64 0.0 0.645 1.29 1.935 ... 895.9 896.6 897.2
│         * y                (y) float64 0.0 0.645 1.29 1.935 ... 668.9 669.5 670.2
│         * z                (z) float64 0.0 2.58 5.16 7.74 ... 451.5 454.1 456.7 459.2
│       Dimensions without coordinates: x_in, x_out
│       Data variables:
│           affine_metadata  (t, x_in, x_out) float64 1.0 0.0 0.0 0.0 ... 0.0 0.0 1.0
│           image            (t, c, z, y, x) uint16 dask.array<chunksize=(1, 1, 179, 256, 256), meta=np.ndarray>
├── DataTree('scale1')
│       Dimensions:          (t: 1, x_in: 4, x_out: 4, c: 1, z: 179, y: 520, x: 696)
│       Coordinates:
│         * c                (c) int64 0
│         * t                (t) int64 0
│         * x                (x) float64 0.3225 1.613 2.902 4.193 ... 894.3 895.6 896.9
│         * y                (y) float64 0.3225 1.613 2.902 4.193 ... 667.3 668.5 669.8
│         * z                (z) float64 0.0 2.58 5.16 7.74 ... 451.5 454.1 456.7 459.2
│       Dimensions without coordinates: x_in, x_out
│       Data variables:
│           affine_metadata  (t, x_in, x_out) float64 1.0 0.0 0.0 0.0 ... 0.0 0.0 1.0
│           image            (t, c, z, y, x) uint16 dask.array<chunksize=(1, 1, 179, 256, 256), meta=np.ndarray>
└── DataTree('scale2')
        Dimensions:          (t: 1, x_in: 4, x_out: 4, c: 1, z: 179, y: 260, x: 348)
        Coordinates:
          * c                (c) int64 0
          * t                (t) int64 0
          * x                (x) float64 0.9675 3.548 6.128 8.707 ... 891.1 893.6 896.2
          * y                (y) float64 0.9675 3.548 6.128 8.707 ... 664.0 666.6 669.2
          * z                (z) float64 0.0 2.58 5.16 7.74 ... 451.5 454.1 456.7 459.2
        Dimensions without coordinates: x_in, x_out
        Data variables:
            affine_metadata  (t, x_in, x_out) float64 1.0 0.0 0.0 0.0 ... 0.0 0.0 1.0
            image            (t, c, z, y, x) uint16 dask.array<chunksize=(1, 1, 179, 256, 256), meta=np.ndarray>
```

## Transformation parameters

### Affine transformation parameters

`xarray.DataArray` containing parameters in the form of
- a homogeneous transform matrix
- of dimensionality (ndim+1, ndim+1)
- datatype `float`

with axis labels
- 't'
- 'x_in'
- 'x_out'

Example string representation:

`print(msims[0]['scale0/affine_manual'])`
```
<xarray.DataArray 'affine_manual' (t: 1, x_in: 4, x_out: 4)>
array([[[1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]]])
Coordinates:
  * t        (t) int64 0
Dimensions without coordinates: x_in, x_out
```
