"""
Declarative, JSON-serialisable descriptions of browser work.

Everything the UI can ask for is expressed as one of these specs. They are the
only vocabulary shared between the page, the persistent session worker and the
compute workers, which lets any worker reconstruct the same Python state from a
message that contains no image data.

Callables are referenced by name through the registries below rather than being
serialised, so no code crosses the JavaScript boundary.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from multiview_stitcher import fusion, registration

#: Pairwise registration functions selectable from the browser.
PAIRWISE_REGISTRATION_FUNCS = {
    "phase_correlation": registration.phase_correlation_registration,
    "marker_based": registration.registration_marker_based,
}

#: Fusion functions selectable from the browser.
FUSION_FUNCS = {
    "weighted_average": fusion.weighted_average_fusion,
    "simple_average": fusion.simple_average_fusion,
    "max": fusion.max_fusion,
}

#: Groupwise parameter resolution methods selectable from the browser.
GROUPWISE_RESOLUTION_METHODS = (
    "global_optimization",
    "shortest_paths",
    "linear_two_pass",
)

#: Pre-registration pruning methods selectable from the browser.
PRUNING_METHODS = (
    None,
    "alternating_pattern",
    "shortest_paths_overlap_weighted",
    "otsu_threshold_on_overlap",
    "keep_axis_aligned",
)


def _lookup(registry, name, what):
    if name not in registry:
        raise ValueError(
            f"Unknown {what} '{name}'. Available: {sorted(registry)}."
        )
    return registry[name]


def _fields_from_dict(cls, payload):
    """Build a dataclass from a dict, ignoring unknown keys."""
    known = set(cls.__dataclass_fields__)
    return cls(
        **{
            key: value
            for key, value in (payload or {}).items()
            if key in known
        }
    )


@dataclass
class SourceSpec:
    """One input OME-Zarr, addressed by a URL the runtime can fetch."""

    url: str
    name: Optional[str] = None

    def resolved_name(self, index=0):
        if self.name:
            return self.name
        trimmed = self.url.rstrip("/").split("/")[-1]
        return trimmed or f"view_{index}"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        if isinstance(payload, str):
            return cls(url=payload)
        return _fields_from_dict(cls, payload)


@dataclass
class RegistrationOptions:
    """Options for :func:`multiview_stitcher.registration.register`."""

    transform_key: Optional[str] = None
    new_transform_key: str = "registered"
    reg_channel_index: Optional[int] = None
    pairwise_reg_func: str = "phase_correlation"
    pairwise_reg_func_kwargs: dict = field(default_factory=dict)
    registration_binning: Optional[dict] = None
    reg_res_level: Optional[int] = None
    overlap_tolerance: Any = 0.0
    groupwise_resolution_method: str = "global_optimization"
    groupwise_resolution_kwargs: dict = field(default_factory=dict)
    pre_registration_pruning_method: Optional[str] = "alternating_pattern"

    def __post_init__(self):
        _lookup(
            PAIRWISE_REGISTRATION_FUNCS,
            self.pairwise_reg_func,
            "pairwise registration function",
        )
        if self.groupwise_resolution_method not in GROUPWISE_RESOLUTION_METHODS:
            raise ValueError(
                f"Unknown groupwise resolution method "
                f"'{self.groupwise_resolution_method}'."
            )
        if self.pre_registration_pruning_method not in PRUNING_METHODS:
            raise ValueError(
                f"Unknown pruning method "
                f"'{self.pre_registration_pruning_method}'."
            )

    def register_kwargs(self):
        """Keyword arguments for `registration.register`, minus the executor."""
        return {
            "transform_key": self.transform_key,
            "new_transform_key": self.new_transform_key,
            "reg_channel_index": self.reg_channel_index,
            "pairwise_reg_func": _lookup(
                PAIRWISE_REGISTRATION_FUNCS,
                self.pairwise_reg_func,
                "pairwise registration function",
            ),
            "pairwise_reg_func_kwargs": dict(self.pairwise_reg_func_kwargs),
            "registration_binning": self.registration_binning,
            "reg_res_level": self.reg_res_level,
            "overlap_tolerance": self.overlap_tolerance,
            "groupwise_resolution_method": self.groupwise_resolution_method,
            "groupwise_resolution_kwargs": dict(
                self.groupwise_resolution_kwargs
            ),
            "pre_registration_pruning_method": (
                self.pre_registration_pruning_method
            ),
        }

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        return _fields_from_dict(cls, payload)


@dataclass
class FusionOptions:
    """Options for :func:`multiview_stitcher.fusion.fuse`."""

    transform_key: Optional[str] = None
    fusion_func: str = "weighted_average"
    output_chunksize: Any = None
    output_spacing: Optional[dict] = None
    output_stack_mode: str = "union"
    output_zarr_url: Optional[str] = None
    ngff_version: str = "0.4"

    def __post_init__(self):
        _lookup(FUSION_FUNCS, self.fusion_func, "fusion function")

    @property
    def is_preview(self):
        """A preview fusion is computed lazily and never written to disk."""
        return self.output_zarr_url is None

    def fuse_kwargs(self):
        kwargs = {
            "transform_key": self.transform_key,
            "fusion_func": _lookup(
                FUSION_FUNCS, self.fusion_func, "fusion function"
            ),
            "output_stack_mode": self.output_stack_mode,
        }
        if self.output_chunksize is not None:
            kwargs["output_chunksize"] = self.output_chunksize
        if self.output_spacing is not None:
            kwargs["output_spacing"] = self.output_spacing
        return kwargs

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        return _fields_from_dict(cls, payload)


@dataclass
class SessionSpec:
    """Everything a worker needs to rebuild a session's Python state.

    `transforms` maps a transform key to one serialised affine per source, so
    registration results reach compute workers without re-registering.
    """

    sources: list = field(default_factory=list)
    transforms: dict = field(default_factory=dict)
    generation: int = 0
    session_id: Optional[str] = None
    #: Options of the fused preview the viewer is currently reading, so that a
    #: compute worker can rebuild the same lazily fused image on demand.
    preview: Optional[dict] = None

    def to_dict(self):
        return {
            "sources": [source.to_dict() for source in self.sources],
            "transforms": self.transforms,
            "generation": int(self.generation),
            "session_id": self.session_id,
            "preview": self.preview,
        }

    @classmethod
    def from_dict(cls, payload):
        if isinstance(payload, cls):
            return payload
        payload = payload or {}
        return cls(
            sources=[
                SourceSpec.from_dict(source)
                for source in payload.get("sources", [])
            ],
            transforms=dict(payload.get("transforms", {})),
            generation=int(payload.get("generation", 0)),
            session_id=payload.get("session_id"),
            preview=payload.get("preview"),
        )
