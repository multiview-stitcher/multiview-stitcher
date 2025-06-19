"""Patch czifile to support ZSTD1 compression in CZI files.
This patch modifies the `SubBlockSegment.data` method to handle ZSTD1 compression
and updates the `DECOMPRESS` dictionary to include a custom decompression function.
See this issue: https://github.com/cgohlke/czifile/issues/10
"""

import typing

import imagecodecs
import numpy
from czifile.czifile import DECOMPRESS, SubBlockSegment, repeat_nd
from scipy.ndimage import zoom


class ZSTD1Header(typing.NamedTuple):
    """
    ZSTD1 header structure
    based on:
    https://github.com/ZEISS/libczi/blob/4a60e22200cbf0c8ff2a59f69a81ef1b2b89bf4f/Src/libCZI/decoder_zstd.cpp#L19
    """

    header_size: int
    hiLoByteUnpackPreprocessing:bool


def parse_zstd1_header(data, size):
    """
    Parse ZSTD header

    https://github.com/ZEISS/libczi/blob/4a60e22200cbf0c8ff2a59f69a81ef1b2b89bf4f/Src/libCZI/decoder_zstd.cpp#L84
    """
    if size < 1:
        return ZSTD1Header(0, False)

    if data[0] == 1:
        return ZSTD1Header(1, False)

    if data[0] == 3 and size < 3:
        return ZSTD1Header(0, False)

    if data[1] == 1:
        return ZSTD1Header(3, bool(data[2] & 1))

    return ZSTD1Header(0, False)


def decode_zstd1(data):
    """
    Decode ZSTD1 data
    """
    header = parse_zstd1_header(data, len(data))
    return imagecodecs.zstd_decode(data[header.header_size:])


# patch data method of class czifile.SubBlockSegment
def data(self, raw=False, resize=True, order=0):
    """Read image data from file and return as numpy array."""
    de = self.directory_entry
    fh = self._fh

    if raw:
        with fh.lock:
            fh.seek(self.data_offset)
            data = fh.read(self.data_size)
        return data
    if de.compression:
        # if de.compression not in DECOMPRESS:
        #     raise ValueError('compression unknown or not supported')
        with fh.lock:
            fh.seek(self.data_offset)
            data = fh.read(self.data_size)
        data = DECOMPRESS[de.compression](data)
        # if de.compression == 2:

        if de.compression in {2, 5, 6}: # patched https://github.com/cgohlke/czifile/issues/10
            # LZW
            data = numpy.fromstring(data, de.dtype)
    else:
        dtype = numpy.dtype(de.dtype)
        with fh.lock:
            fh.seek(self.data_offset)
            data = fh.read_array(dtype, self.data_size // dtype.itemsize)

    data = data.reshape(de.stored_shape)
    if de.compression != 4 and de.stored_shape[-1] in (3, 4):
        if de.stored_shape[-1] == 3:
            # BGR -> RGB
            data = data[..., ::-1]
        else:
            # BGRA -> RGBA
            tmp = data[..., 0].copy()
            data[..., 0] = data[..., 2]
            data[..., 2] = tmp
    if de.stored_shape == de.shape or not resize:
        return data

    # sub / supersampling
    factors = [j / i for i, j in zip(de.stored_shape, de.shape)]
    factors = [(int(round(f)) if abs(f - round(f)) < 0.0001 else f)
                for f in factors]

    # use repeat if possible
    if order == 0 and all(isinstance(f, int) for f in factors):
        data = repeat_nd(data, factors).copy()
        data.shape = de.shape
        return data

    # remove leading dimensions with size 1 for speed
    shape = list(de.stored_shape)
    i = 0
    for s in shape:
        if s != 1:
            break
        i += 1
    shape = shape[i:]
    factors = factors[i:]
    data.shape = shape

    # resize RGB components separately for speed
    if zoom is None:
        raise ImportError("cannot import 'zoom' from scipy or ndimage")
    if shape[-1] in (3, 4) and factors[-1] == 1.0:
        factors = factors[:-1]
        old = data
        data = numpy.empty(de.shape, de.dtype[-2:])
        for i in range(shape[-1]):
            data[..., i] = zoom(old[..., i], zoom=factors, order=order)
    else:
        data = zoom(data, zoom=factors, order=order)

    data.shape = de.shape
    return data


# patch DECOMPRESS dictionary and decompression function
DECOMPRESS[6] = decode_zstd1
SubBlockSegment.data = data
