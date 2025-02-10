# Accuracy

Here are some things to do if you encounter issues with the accuracy of the registration.


## Determining which pairs of tiles/views are registered

Do you have a regular grid of input tiles? In that case it's recommended to use `register(..., pre_registration_pruning="keep_axis_aligned")` available in multiview-stitcher version>=0.18, which disregards diagonal overlaps during pairwise registration.

```python
register(
  ...,
  pre_registration_pruning="keep_axis_aligned", # None, 'keep_axis_aligned', 'shortest_paths_overlap_weighted'
  )
```

## Registration binning

1. Decrease the registration binning, with {'z': 1, 'y': 1, 'x': 1} performing the registration on the highest possible resolution.

!!! note "Binning"
    Keep in mind that typically not much is gained from binning factors lower than 2.

## Choosing the right method for pairwise registration

### Type of transform

Consider which transform suits your data best:
1. Translation
1. Rigid
1. Similarity
1. Affine

### Available registration implementations:
1. Phase correlation (translation)
1. AntsP (translation, rigid, similarity, affine)
1. ITK-elastix (translation, rigid, similarity, affine) (WIP)
1. [Custom registration functions](extension_api_pairwise_registration.md) (up to affine)

## Change the parameters used during global parameter resolution

### Type of transform

Following pairwise registration, determining the global transformation parameters can also be done assuming different types of transforms:
  1. Translation
  1. Rigid
  1. Similarity
  1. Affine

Example:
```python
registration.register(
  ...
  groupwise_resolution_kwargs={
      'transform': 'translation', # 'rigid', 'similarity', 'affine'
  }
)
```

### Alternative: simple global parameter resolution

```python
registration.register(
  ...
  groupwise_resolution_method='shortest_paths',
  groupwise_resolution_kwargs={
      # no options available
  }
)
```
