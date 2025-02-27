# Pairwise registration

Custom functions for pairwise registration can be passed to the `registration.register` function:

```python
def register(
    ...
    transform_key,
    pairwise_reg_func: Callable = phase_correlation_registration,
    pairwise_reg_func_kwargs: dict = None,
    ...
)
```

Custom registration functions can have one of the following two signatures. `registration.register` will automatically detect which signature is used and call the function accordingly, passing to the function
- `pairwise_reg_func_kwargs`
- the below detailed parameters to the functions.

## Registration in pixel space

This API is for adding registration functions that operate on pixel data without any knowledge of the physical space.

!!! note "Initial transformation"
    The moving data array is pre-transformed to match the pixel coordinate space of the fixed data array, using the affine matrices of the `transform_key` passed to `registration.register`. Image data is passed as float dtype and NaN values mark pixels that map outside the image domains after transformation.

!!! note "Expected output transformation"
    The affine matrix returned by a registration following this API transforms pixel indices of the fixed image to pixel indices of the moving image. `registration.register` will take care of converting this matrix to a transformation that transforms the physical space.

```python
def pairwise_registration(
    fixed_data: Array-like[np.float32],
    moving_data: Array-like[np.float32],
    **kwargs, # additional keyword arguments passed `pairwise_reg_func_kwargs`
    ) -> dict:

    # Registration code here

    return {
        "affine_matrix": affine_matrix, # homogenous matrix of shape (ndim + 1, ndim + 1), axis order (z, y, x)
        "quality": , # float between 0 and 1 (if not available, set to 1.0)
    }
```

Note that in the case of pixel space registration, the input data is passed as `np.float32` arrays (as opposed to registration in physical coordinate space, during which the dtype remains unaltered). Invalid pixel values, i.e. those that map outside of the input data, are marked with NaNs.

Example implementation: `multiview_stitcher.registration.phase_correlation_registration` (default pairwise registration function).

## Registration in physical coordinate space

This API is for adding registration functions that operate in physical coordinates.

!!! note "Transformations in physical space"
    Both the pre-calculated `initial_affine` parameter passed to the pairwise registration function and the affine matrix returned by a registration following this API transform physical positions of the fixed image to physical positions of the moving image. Here, physical positions are calculated as `origin + spacing * pixel_index`.

```python
def custom_pairwise_registration_function(
    fixed_data: Array-like,
    moving_data: Array-like,
    *,
    fixed_origin: dict[str, float], # e.g. {"z": 10.0, "y": 20.0, "x": 30.0}
    moving_origin: dict[str, float],
    fixed_spacing: dict[str, float], # e.g. {"z": 1.0, "y": 0.5, "x": 0.5}
    moving_spacing: dict[str, float],
    initial_affine: xr.DataArray, # see note below
    **kwargs, # `pairwise_reg_func_kwargs` passed to `registration.register`
) -> dict:
    # Registration code here

    return {
        "affine_matrix": affine_matrix, # homogenous matrix of shape (ndim + 1, ndim + 1), axis order (z, y, x)
        "quality": 1.0, # float between 0 and 1 (if not available, set to 1.0)
    }
```

For a description of the object that will be passed as an initial affine matrix, see [here](objects.md#affine-transformation-parameters).

## Fusion

Custom functions can be passed to the `fusion.fuse` function. `multiview-stitcher` provides the custom fusion and weights functions with pre-transformed chunks corresponding to the `transform_key` passed to `fusion.fuse`. Only those views will be passed which contribute to the given output chunks. The elements of the input lists correspond to the different input views / tiles and conserve the order of the input views / tiles passed to `fusion.fuse`.

```python
def fuse(
    ...
    transform_key: str = None,
    fusion_func: Callable = weighted_average_fusion,
    fusion_func_kwargs: dict = None,
    weights_func: Callable = None, # by default no additional fusion weights are used
    weights_func_kwargs: dict = None,
    ...
```

### Custom fusion methods

Custom function for fusion of pre-transformed view chunks. This could implement e.g. a maximum intensity projection, a weighted average, or a multi-view deconvolution.

```python
def custom_fusion_function(
    transformed_views: List[Array-like], # list of pre-transformed view chunks
    blending_weights: List[Array-like], # optional functional argument
    fusion_weights: List[Array-like], # optional functional argument
    **kwargs, # `fusion_func_kwargs` passed to `fusion.fuse`
) -> Array-like:

    # Fusion code here

    return fused_array
```

If the optional funtion arguments are not part of the function signature, the arguments will be ignored.

Example implementation: `multiview_stitcher.fusion.weighted_average_fusion`.

### Custom weight functions

Custom function for calculating additional fusion weights passed to the fusion function as `fusion_weights`. This could implement e.g. a content-based weighting scheme. The function should return a list of weights, one for each view chunk.

```python
def custom_weight_function(
    transformed_views : List[Array-like],
    blending_weights : List[Array-like],
    **kwargs, # `weights_func_kwargs` passed to `fusion.fuse`
) - > List[Array-like]:

    # Weight calculation code here

    return weights
```

Example implementation: `multiview_stitcher.weights.content_based`.

## Global parameter resolution

Custom function API to be added.
