# Fusion

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

## Custom fusion methods

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

## Custom weight functions

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
