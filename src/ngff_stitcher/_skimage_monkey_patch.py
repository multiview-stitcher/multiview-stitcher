# skimage function modified to consider masks

"""
Monkey-patch skimage.registration._phase_cross_correlation._disambiguate_shift
to consider masks when comparing shift candidates. This showed better performance
on a 2d stitching dataset by Arthur Michaut. More investigations needs to be
done, also to justify proposing a PR to skimage.
"""

import itertools

import numpy as np
from scipy import ndimage as ndi


# modified from skimage
def _modified_disambiguate_shift(reference_image, moving_image, shift):
    """Determine the correct real-space shift based on periodic shift.

    When determining a translation shift from phase cross-correlation in
    Fourier space, the shift is only correct to within a period of the image
    size along each axis, resulting in $2^n$ possible shifts, where $n$ is the
    number of dimensions of the image. This function checks the
    cross-correlation in real space for each of those shifts, and returns the
    one with the highest cross-correlation.

    The strategy we use is to perform the shift on the moving image *using the
    'grid-wrap' mode* in `scipy.ndimage`. The moving image's original borders
    then define $2^n$ quadrants, which we cross-correlate with the reference
    image in turn using slicing. The entire operation is thus $O(2^n + m)$,
    where $m$ is the number of pixels in the image (and typically dominates).

    Parameters
    ----------
    reference_image : numpy array
        The reference (non-moving) image.
    moving_image : numpy array
        The moving image: applying the shift to this image overlays it on the
        reference image. Must be the same shape as the reference image.
    shift : tuple of float
        The shift to apply to each axis of the moving image, *modulo* image
        size. The length of ``shift`` must be equal to ``moving_image.ndim``.

    Returns
    -------
    real_shift : tuple of float
        The shift disambiguated in real space.
    """
    shape = reference_image.shape
    positive_shift = [shift_i % s for shift_i, s in zip(shift, shape)]
    negative_shift = [shift_i - s for shift_i, s in zip(positive_shift, shape)]
    subpixel = np.any(np.array(shift) % 1 != 0)
    interp_order = 3 if subpixel else 0
    shifted = ndi.shift(
        # moving_image, shift, mode='grid-wrap', order=interp_order,
        moving_image.astype(float),
        shift,
        mode="constant",
        order=interp_order,
        cval=np.nan,
    )
    indices = np.round(positive_shift).astype(int)
    splits_per_dim = [(slice(0, i), slice(i, None)) for i in indices]
    max_corr = -1.0
    max_slice = None
    for test_slice in itertools.product(*splits_per_dim):
        mask = ~np.isnan(shifted[test_slice])
        reference_tile = np.reshape(reference_image[test_slice][mask], -1)
        moving_tile = np.reshape(shifted[test_slice][mask], -1)
        # reference_tile = np.reshape(reference_image[test_slice], -1)
        # moving_tile = np.reshape(shifted[test_slice], -1)
        corr = -1.0
        if reference_tile.size > 2:
            corr = np.corrcoef(reference_tile, moving_tile)[0, 1]
            # corr = skimage.metrics.normalized_mutual_information(reference_tile, moving_tile)
        if corr > max_corr:
            max_corr = corr
            max_slice = test_slice
    real_shift_acc = []
    for sl, pos_shift, neg_shift in zip(
        max_slice, positive_shift, negative_shift
    ):
        real_shift_acc.append(pos_shift if sl.stop is None else neg_shift)
    if not subpixel:
        real_shift = tuple(map(int, real_shift_acc))
    else:
        real_shift = tuple(real_shift_acc)
    return real_shift
