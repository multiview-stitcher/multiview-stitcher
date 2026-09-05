"""OME-Zarr 0.6 metadata and static spatial affine adapters.

Pixel calibration stays on each dataset; registration maps intrinsic physical
coordinates to a named output system. No pixel I/O or resampling happens here.
"""

from copy import deepcopy
from dataclasses import asdict

import numpy as np

from multiview_stitcher import spatial_image_utils as si_utils


def transform_matrix(transform, ndim):
    """Evaluate an inline, dimension-preserving linear NGFF transformation."""
    tf = asdict(transform) if not isinstance(transform, dict) else transform
    kind = tf.get("type")
    matrix = np.eye(ndim + 1)
    if tf.get("path") is not None:
        raise NotImplementedError(
            "Array-backed NGFF transforms are unsupported."
        )
    if kind == "sequence":
        for child in tf["transformations"]:
            if child.get("input") or child.get("output"):
                raise NotImplementedError(
                    "Named intermediate systems in sequences are unsupported."
                )
            matrix = transform_matrix(child, ndim) @ matrix
    elif kind == "identity":
        pass
    elif kind in ("scale", "translation", "rotation", "affine"):
        values = np.asarray(tf[kind], dtype=float)
        expected = {
            "scale": (ndim,),
            "translation": (ndim,),
            "rotation": (ndim, ndim),
            "affine": (ndim, ndim + 1),
        }[kind]
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(
                f"Invalid {kind} parameters; expected {expected}."
            )
        if kind == "scale":
            matrix[np.arange(ndim), np.arange(ndim)] = values
        elif kind == "translation":
            matrix[:-1, -1] = values
        elif kind == "rotation":
            matrix[:-1, :-1] = values
        else:
            matrix[:-1] = values
    else:
        raise NotImplementedError(
            f"Unsupported NGFF transform type: {kind!r}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("NGFF transform composition is not finite.")
    return matrix


def static_affine(sim, transform_key):
    """Extract one physical-space affine, rejecting varying t/c transforms."""
    ndim = len(si_utils.get_spatial_dims_from_sim(sim))
    if transform_key is None:
        return np.eye(ndim + 1)
    affine = si_utils.get_affine_from_sim(sim, transform_key)
    other_dims = [d for d in affine.dims if d not in ("x_in", "x_out")]
    values = np.asarray(affine.transpose(*other_dims, "x_in", "x_out"))
    matrices = values.reshape((-1, ndim + 1, ndim + 1))
    if not len(matrices) or not np.isfinite(matrices).all():
        raise ValueError("Registration must contain finite affine matrices.")
    if not np.allclose(matrices, matrices[0], rtol=0, atol=1e-12):
        raise NotImplementedError(
            "OME-Zarr export supports only static registration; "
            "select a timepoint/channel before exporting varying transforms."
        )
    matrix = matrices[0]
    if not np.allclose(matrix[-1], np.eye(ndim + 1)[-1]):
        raise ValueError("Registration must be a homogeneous affine matrix.")
    return matrix


def registration_transform(axes, affine, intrinsic, target):
    """Embed a spatial affine into the full on-disk axis order."""
    from ngff_zarr.v06.zarr_metadata import Affine, CoordinateSystemIdentifier

    if not target or target == intrinsic:
        raise ValueError("Registration target must differ from intrinsic.")
    spatial = [i for i, ax in enumerate(axes) if ax["type"] == "space"]
    indices = spatial + [len(axes)]
    matrix = np.eye(len(axes) + 1)
    if affine.shape != (len(indices), len(indices)):
        raise ValueError("Affine rank does not match the spatial axes.")
    matrix[np.ix_(indices, indices)] = affine
    return Affine(
        affine=matrix[:-1].tolist(),
        input=CoordinateSystemIdentifier(name=intrinsic),
        output=CoordinateSystemIdentifier(name=target),
    )


def build_metadata(axes, datasets, name, affine=None, target="registered"):
    """Build metadata with ngff-zarr's version-specific data model."""
    from ngff_zarr.v06.zarr_metadata import (
        Axis,
        CoordinateSystem,
        CoordinateSystemIdentifier,
        Dataset,
        Identity,
        Metadata,
        Scale,
        TransformSequence,
        Translation,
    )

    systems = [
        CoordinateSystem(name="intrinsic", axes=[Axis(**a) for a in axes])
    ]
    levels = []
    for dataset in datasets:
        transforms = []
        for tf in dataset["coordinateTransformations"]:
            kind = tf["type"]
            if kind == "scale":
                transforms.append(Scale(scale=tf["scale"]))
            elif kind == "translation":
                transforms.append(Translation(translation=tf["translation"]))
            elif kind == "identity":
                transforms.append(Identity())
            else:
                raise ValueError(f"Invalid dataset calibration: {kind}.")
        if len(transforms) == 1 and transforms[0].type in (
            "scale",
            "identity",
        ):
            calibration = transforms[0]
            calibration.input = CoordinateSystemIdentifier(
                path=dataset["path"]
            )
            calibration.output = CoordinateSystemIdentifier(name="intrinsic")
        else:
            calibration = TransformSequence(
                transformations=transforms,
                input=CoordinateSystemIdentifier(path=dataset["path"]),
                output=CoordinateSystemIdentifier(name="intrinsic"),
            )
        levels.append(
            Dataset(
                path=dataset["path"], coordinateTransformations=[calibration]
            )
        )
    registrations = None
    if affine is not None:
        registrations = [
            registration_transform(axes, affine, "intrinsic", target)
        ]
        systems.append(
            CoordinateSystem(name=target, axes=deepcopy(systems[0].axes))
        )
    metadata = Metadata(
        coordinateSystems=systems,
        datasets=levels,
        coordinateTransformations=registrations,
        name=name,
    )
    validate_calibration(metadata)
    return metadata


def validate_calibration(metadata):
    """Reject calibrations that cannot be represented by sim coordinates."""
    intrinsic = metadata.intrinsic_coordinate_system
    axes = intrinsic.axes
    names = [ax.name for ax in axes]
    spatial = [ax.name for ax in axes if ax.type == "space"]
    if spatial not in (["y", "x"], ["z", "y", "x"]):
        raise NotImplementedError("NGFF spatial axes must be y,x or z,y,x.")
    expected = [d for d in ("t", "c", "z", "y", "x") if d in names]
    if names != expected or len(set(names)) != len(names):
        raise NotImplementedError("Unsupported NGFF axis names or order.")
    for ax in axes:
        if ax.type != {"t": "time", "c": "channel"}.get(ax.name, "space"):
            raise NotImplementedError("Unsupported NGFF axis type.")
    for ds in metadata.datasets:
        if len(ds.coordinateTransformations) != 1:
            raise ValueError(
                "Each 0.6 dataset needs one calibration transform."
            )
        tf = ds.coordinateTransformations[0]
        if tf.input is None or tf.input.path != ds.path:
            raise ValueError(
                "Dataset transform input must match its array path."
            )
        if (
            tf.output is None
            or tf.output.name != intrinsic.name
            or tf.output.path
        ):
            raise ValueError(
                "Dataset output must be the local intrinsic system."
            )
        if tf.type == "sequence":
            if [t.type for t in tf.transformations] != [
                "scale",
                "translation",
            ]:
                raise ValueError(
                    "Dataset sequence must be scale then translation."
                )
        elif tf.type not in ("scale", "identity"):
            raise ValueError("Dataset calibration must be scale or identity.")
        matrix = transform_matrix(tf, len(axes))
        if np.any(np.diag(matrix) <= 0):
            raise ValueError("Dataset scale must be positive.")
        if "c" in names:
            c = names.index("c")
            if matrix[c, c] != 1 or matrix[c, -1] != 0:
                raise NotImplementedError(
                    "Channel calibration must be identity."
                )


def spatial_affine(metadata, target=None):
    """Resolve a direct intrinsic/target mapping; never guess among targets.

    Explicitly selecting the intrinsic system opts out of registration.
    Cross-image scenes and graph traversal are intentionally separate work.
    """
    intrinsic = metadata.intrinsic_coordinate_system
    axes = intrinsic.axes
    systems = {cs.name: cs for cs in metadata.coordinateSystems}
    if len(systems) != len(metadata.coordinateSystems):
        raise ValueError("Coordinate system names must be unique.")
    spatial = [i for i, ax in enumerate(axes) if ax.type == "space"]
    identity = np.eye(len(spatial) + 1)
    if target == intrinsic.name:
        return identity
    transforms = metadata.coordinateTransformations or []
    candidates = []
    for tf in transforms:
        if tf.input is None or tf.output is None:
            raise ValueError(
                "Registration needs named input and output systems."
            )
        if tf.input.path or tf.output.path:
            # Label links are not image registrations. Other cross-image links
            # require a scene-aware reader rather than silently losing placement.
            paths = [p for p in (tf.input.path, tf.output.path) if p]
            if all(p.startswith("labels/") for p in paths):
                continue
            raise NotImplementedError(
                "Cross-image NGFF transforms are unsupported."
            )
        if tf.input.name == intrinsic.name:
            candidates.append((tf.output.name, tf, False))
        elif tf.output.name == intrinsic.name:
            candidates.append((tf.input.name, tf, True))
        else:
            raise NotImplementedError(
                "Indirect NGFF registration paths are unsupported."
            )
    if target is None:
        if not candidates:
            return identity
        if len(candidates) != 1:
            raise ValueError(
                "Multiple registrations; specify target_coordinate_system."
            )
        target = candidates[0][0]
    matches = [entry for entry in candidates if entry[0] == target]
    if len(matches) != 1 or target not in systems:
        raise ValueError(
            f"No unique registration to coordinate system {target!r}."
        )
    # Units and axis order define the numeric basis of the affine. Require the
    # same basis for now instead of implicitly permuting or converting units.
    if [(a.name, a.type, a.unit) for a in axes] != [
        (a.name, a.type, a.unit) for a in systems[target].axes
    ]:
        raise NotImplementedError(
            "Registration axes and units must match intrinsic."
        )
    _, tf, reverse = matches[0]
    matrix = transform_matrix(tf, len(axes))
    if reverse:
        matrix = np.linalg.inv(matrix)
    nonspatial = [i for i in range(len(axes)) if i not in spatial]
    full_identity = np.eye(len(axes) + 1)
    if not (
        np.allclose(matrix[nonspatial], full_identity[nonspatial])
        and np.allclose(matrix[:, nonspatial], full_identity[:, nonspatial])
    ):
        raise NotImplementedError(
            "Time/channel transforms or axis mixing are unsupported."
        )
    indices = spatial + [len(axes)]
    return matrix[np.ix_(indices, indices)]
