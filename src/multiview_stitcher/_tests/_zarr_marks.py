"""Shared pytest marks for functionality that needs zarr-python v3.

multiview-stitcher's core - NGFF 0.4 reading and writing, registration, fusion
- works under both zarr-python v2 and v3, so that it can be installed next to
packages that still pin ``zarr<3``. A few features are built directly on the v3
API and have no v2 equivalent:

* the virtual-array layer in :mod:`multiview_stitcher.zarr_utils`, and with it
  zarr-backed sims (lazy singleton expansion, lazy concat/stack),
* TIFF files read through a virtual zarr store,
* OME-Zarr 0.5 (a zarr v3 hierarchy),
* the browser/Pyodide runtime.

Their tests carry :data:`zarr_v3_only`. Everything that is *not* marked is
expected to pass under either library version - that is what the ``zarr2`` tox
environment checks.
"""

import pytest

from multiview_stitcher._zarr_compat import ZARR_V3

zarr_v3_only = pytest.mark.skipif(
    not ZARR_V3,
    reason="requires zarr-python >= 3",
)

#: ``array_backend`` parametrization for readers that can hand back either a
#: zarr-backed or a dask-backed sim. Under zarr v2 the zarr backend degrades to
#: dask (with a warning), so only the dask case is meaningful there.
ARRAY_BACKENDS = [
    pytest.param("zarr", marks=zarr_v3_only),
    "dask",
]
