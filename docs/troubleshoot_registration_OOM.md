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

By default, the registration of all the pairs to be registered is run in parallel. Therefore reducing the parallelism can help. This can be done in different ways:

1. Setting `register(..., scheduler='single-threaded')` will perform one pairwise registration at a time (simplest).

1. Explicitly setting the amount of threads using a dask distributed cluster. How does this work? Once a dask distributed client object exists, dask will use it as a computational backend, even if the client object is not passed to the `register` function. In the context of multiview-stitcher, we recommend to create a cluster with one worker and multiple threads per worker. This can be done as follows:

```python
from distributed import Client, LocalCluster
local_cluster = LocalCluster(n_workers=1, threads_per_worker=4)
client = Client(local_cluster)
```
