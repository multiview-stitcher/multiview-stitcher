"""
Executors that spread registration and fusion over a pool of web workers.

Both follow the same shape: the session worker describes the work as JSON,
blocks on the bridge while the pool runs it, and merges the results back into
the ordinary in-process code path. The heavy objects (images, dask graphs,
zarr stores) are rebuilt inside each worker from the session spec and never
serialised.
"""

import math

import xarray as xr

from multiview_stitcher import msi_utils
from multiview_stitcher.browser import serialization
from multiview_stitcher.browser.bridge import get_bridge
from multiview_stitcher.browser.specs import (
    PAIRWISE_REGISTRATION_FUNCS,
    FusionOptions,
)


def n_timepoints(msim):
    """How many timepoints a view has, or None if it has no time axis.

    None and 1 are deliberately different: a view without a time axis must be
    registered as it is, since selecting a timepoint on it would fail, while a
    single-timepoint view is selected from like any other.
    """
    sim = msi_utils.get_sim_from_msim(msim)
    return int(sim.sizes["t"]) if "t" in sim.dims else None


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


def concat_over_time(parts):
    """Join per-timepoint pairwise results into one result over time.

    Each part is what one call of the pairwise registration function produced,
    already carrying the timepoint it was computed for as a coordinate. Joining
    them here is what lets the timepoints of one pair be registered on
    different workers while `register` still receives a single array per edge.
    """
    if len(parts) == 1:
        return parts[0]

    return {
        key: xr.concat([part[key] for part in parts], dim="t")
        for key in ("transform", "quality", "bbox")
    }


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

    One task is one call of the pairwise registration function: a pair of views
    at one timepoint. That is the smallest piece of work there is, so the pool
    stays balanced however unevenly the pairs are matched, and it is the unit
    the progress bar counts - a timelapse otherwise shows nothing at all until
    a whole pair, every timepoint of it, is finished.
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
        n_t = n_timepoints(msims[0])

        # One pair per task by default: the pool queues tasks over its workers,
        # which balances the load even when pairs differ a lot in cost.
        groups = [
            list(edges[i : i + self.max_pairs_per_task])
            for i in range(0, len(edges), self.max_pairs_per_task)
        ]

        # Timepoints are addressed by index rather than by coordinate value:
        # the index survives JSON whatever the coordinate is made of, and the
        # worker reads the values off the view it rebuilt.
        time_slices = [None] if n_t is None else [[t] for t in range(n_t)]

        tasks = [
            {
                "kind": "register_pairs",
                "edges": [[int(a), int(b)] for a, b in group],
                "register_kwargs": options,
                "reg_channel": reg_channel,
                "time_indices": indices,
                "units": len(group),
            }
            for indices in time_slices
            for group in groups
        ]

        # With one timepoint a task is a pair and the bar can say so. Over a
        # timelapse it is one pair at one timepoint, and naming both is the
        # difference between a bar that looks stuck and one that explains why
        # there is so much of it.
        over_time = len(time_slices) > 1
        progress = {
            "label": "registering",
            "unit": "registration" if over_time else "pair",
        }
        if over_time:
            progress["detail"] = f"{len(edges)} pairs × {n_t} timepoints"

        results = self.bridge.dispatch(tasks, session=spec, progress=progress)

        pairwise = [
            serialization.pairwise_result_from_json(item)
            for result in results
            for item in result["pairwise"]
        ]

        expected = len(edges) * len(time_slices)
        if len(pairwise) != expected:
            raise RuntimeError(
                f"Worker pool returned {len(pairwise)} pairwise results for "
                f"{len(edges)} pairs over {len(time_slices)} timepoint(s)."
            )

        # Tasks were laid out timepoint by timepoint, each covering every edge
        # in order, so one edge's timepoints are `len(edges)` apart. Joining
        # them gives back the one array over time `register` resolves from.
        return [
            concat_over_time(pairwise[edge :: len(edges)])
            for edge in range(len(edges))
        ]


class RemoteFusionExecutor:
    """Fuse the blocks of a Zarr output across the worker pool."""

    #: Blocks per task. Small tasks keep the pool balanced and the progress
    #: bar moving; the cost of one more task is a few hundred bytes of JSON.
    blocks_per_task = 4

    def __init__(self, session_spec, bridge=None):
        self.session_spec = session_spec
        self.bridge = bridge or get_bridge()

    def __call__(self, options, levels):
        """Fuse every block of every level across the pool.

        Blocks are split so that each task writes a disjoint set of chunk
        files, which is what makes concurrent writes to one output directory
        safe: no two workers ever open the same file.
        """
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

        tasks = []
        for level in levels:
            ids = list(level["block_ids"])
            for start in range(0, len(ids), self.blocks_per_task):
                tasks.append(
                    {
                        "kind": "fuse_blocks",
                        "options": options_payload,
                        "level": level["level"],
                        "block_ids": ids[start : start + self.blocks_per_task],
                        # Progress counts blocks, not tasks, so a bar advances
                        # evenly regardless of how work is grouped.
                        "units": len(ids[start : start + self.blocks_per_task]),
                    }
                )

        results = self.bridge.dispatch(
            tasks,
            session=spec,
            progress={"label": "fusing", "unit": "block"},
        )
        return sum(int(result.get("n_blocks", 0)) for result in results)
