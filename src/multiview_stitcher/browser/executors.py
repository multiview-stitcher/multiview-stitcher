"""
Executors that spread registration and fusion over a pool of web workers.

Both follow the same shape: the session worker describes the work as JSON,
blocks on the bridge while the pool runs it, and merges the results back into
the ordinary in-process code path. The heavy objects (images, dask graphs,
zarr stores) are rebuilt inside each worker from the session spec and never
serialised.
"""

import math

from multiview_stitcher import msi_utils
from multiview_stitcher.browser import serialization
from multiview_stitcher.browser.bridge import get_bridge
from multiview_stitcher.browser.specs import (
    PAIRWISE_REGISTRATION_FUNCS,
    FusionOptions,
)


def selected_channel(msim):
    """The channel a view has already been reduced to, or None.

    `registration.register` selects the registration channel *before* handing
    the views to the pairwise step, so the executor can read the selection off
    the views it is given instead of being told about it separately. Deriving
    it here means the two can never disagree.
    """
    sim = msi_utils.get_sim_from_msim(msim)
    if "c" not in sim.coords or "c" in sim.dims:
        return None
    return serialization.to_jsonable(sim.coords["c"].values)


def _name_of_pairwise_reg_func(func):
    for name, candidate in PAIRWISE_REGISTRATION_FUNCS.items():
        if candidate is func:
            return name
    raise ValueError(
        f"Pairwise registration function {func!r} cannot be dispatched to "
        "workers; it is not one of "
        f"{sorted(PAIRWISE_REGISTRATION_FUNCS)}."
    )


def serialize_register_kwargs(register_kwargs):
    """Make the kwargs `compute_pairwise_registrations` passes down JSON-safe."""
    payload = dict(register_kwargs)
    payload["pairwise_reg_func"] = _name_of_pairwise_reg_func(
        payload["pairwise_reg_func"]
    )
    return serialization.to_jsonable(payload)


def deserialize_register_kwargs(payload):
    """Inverse of :func:`serialize_register_kwargs`."""
    kwargs = dict(payload)
    kwargs["pairwise_reg_func"] = PAIRWISE_REGISTRATION_FUNCS[
        kwargs["pairwise_reg_func"]
    ]
    return kwargs


def split_evenly(items, n_parts):
    """Split ``items`` into at most ``n_parts`` contiguous, near-equal parts."""
    items = list(items)
    n_parts = max(1, min(int(n_parts), len(items))) if items else 0
    if not n_parts:
        return []

    size = math.ceil(len(items) / n_parts)
    return [items[i : i + size] for i in range(0, len(items), size)]


class RemotePairwiseExecutor:
    """`pairwise_executor` for `registration.register` backed by a worker pool.

    Each pairwise registration becomes one task, which lets the pool keep every
    worker busy even when the pairs differ a lot in cost.
    """

    def __init__(self, session_spec, bridge=None, max_pairs_per_task=1):
        self.session_spec = session_spec
        self.bridge = bridge or get_bridge()
        self.max_pairs_per_task = max(1, int(max_pairs_per_task))

    def __call__(self, msims, edges, register_kwargs):
        if self.bridge is None:
            raise RuntimeError(
                "No bridge is installed; cannot dispatch registrations to "
                "workers."
            )

        if not edges:
            return []

        spec = (
            self.session_spec.to_dict()
            if hasattr(self.session_spec, "to_dict")
            else self.session_spec
        )
        options = serialize_register_kwargs(register_kwargs)
        # Workers rebuild full views from the spec, so they have to repeat the
        # channel selection that `register` already applied to `msims`.
        reg_channel = selected_channel(msims[0])

        # One task per pair by default: the pool queues tasks over its workers,
        # which balances the load even when pairs differ a lot in cost.
        groups = [
            list(edges[i : i + self.max_pairs_per_task])
            for i in range(0, len(edges), self.max_pairs_per_task)
        ]

        tasks = [
            {
                "kind": "register_pairs",
                "session": spec,
                "edges": [[int(a), int(b)] for a, b in group],
                "register_kwargs": options,
                "reg_channel": reg_channel,
            }
            for group in groups
        ]

        results = self.bridge.dispatch(tasks)

        pairwise = []
        for result in results:
            pairwise += [
                serialization.pairwise_result_from_json(item)
                for item in result["pairwise"]
            ]

        if len(pairwise) != len(edges):
            raise RuntimeError(
                f"Worker pool returned {len(pairwise)} pairwise results for "
                f"{len(edges)} pairs."
            )

        return pairwise


class RemoteFusionExecutor:
    """Fuse the blocks of a Zarr output across the worker pool."""

    def __init__(self, session_spec, bridge=None, n_workers=None):
        self.session_spec = session_spec
        self.bridge = bridge or get_bridge()
        self.n_workers = n_workers or 1

    def __call__(self, options, block_ids):
        if self.bridge is None:
            raise RuntimeError(
                "No bridge is installed; cannot dispatch fusion blocks to "
                "workers."
            )

        spec = (
            self.session_spec.to_dict()
            if hasattr(self.session_spec, "to_dict")
            else self.session_spec
        )
        options_payload = (
            options.to_dict()
            if isinstance(options, FusionOptions)
            else dict(options)
        )

        groups = split_evenly(block_ids, self.n_workers)
        tasks = [
            {
                "kind": "fuse_blocks",
                "session": spec,
                "options": options_payload,
                "block_ids": group,
            }
            for group in groups
        ]

        results = self.bridge.dispatch(tasks)
        return sum(int(result.get("n_blocks", 0)) for result in results)
