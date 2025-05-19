# MemoryError

Here are some things to do if you encounter a `MemoryError` during registration.

## Increase registration binning

1. Increase the registration binning e.g. to sth like `{'z': 2, 'y': 4, 'x': 4}`. Aim for factors that lead to isotropic and downsampled spacing for registration. Probably even a factor of ~4 can produce accurate subpixel registration parameters (with respect to binning 1).

```python
register(
  ...,
  registration_binning={'z': 2, 'y': 4, 'x': 4},
  )
```

## Decrease computation parallelism

By default, the registration of all the pairs to be registered is run in parallel. Therefore reducing the parallelism can help. This can be done by setting `register(..., n_parallel_pairwise_regs=4)`, which in this example will limit the pairwise registrations that are run in parallel to four.

!!! note "Estimating memory requirements"
    The memory required for each pairwise registration can be estimated by considering that the overlapping image regions need to be loaded in memory (taking into consideration the binning factors), multiplied by a factor of approx. 2-4 for the registration process (depending on the registration function used).
