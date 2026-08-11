"""
Pairwise registration with elastix, on either build of it.

:func:`registration_ITKElastix` is a `pairwise_reg_func` for
:func:`multiview_stitcher.registration.register`, which is also where it is
importable from. Everything else in this module is what it needs.

elastix comes in two builds, and this runs on both:

* `itk-elastix`, a native extension, which is what the desktop package
  installs, and
* `itkwasm-elastix`, the same elastix compiled to WebAssembly. It is the only
  one that exists in the browser, and its backend lives in
  :mod:`multiview_stitcher.browser.elastix`.

They differ in three things: how an image is made, where a transform's
defaults come from, and how one stage is run. Those three are what a *backend*
provides. Everything that decides the registration - which stages run, the
parameter maps they get, the transform elastix starts from and how the result
is read back - is here and shared, so a change to any of it takes effect in
every runtime at once.

What the two exchange is elastix's *parameter object*: a list of parameter
maps, one per transform, oldest first. It is the initial transform on the way
in and the accumulated chain on the way out, it is plain JSON either way, and
nothing has to be written to disk - which matters in a browser, where the
filesystem is a virtual one inside a worker.
"""

import numpy as np

from multiview_stitcher import param_utils

#: Transform stages :func:`registration_ITKElastix` can run, as
#: ``{name: (elastix default parameter map, elastix transform)}``. The default
#: map only supplies defaults - metric, optimizer, pyramid - so a similarity
#: transform can borrow the rigid one and swap the transform itself.
TRANSFORM_TYPES = {
    "translation": ("translation", "TranslationTransform"),
    "rigid": ("rigid", "EulerTransform"),
    "similarity": ("rigid", "SimilarityTransform"),
    "affine": ("affine", "AffineTransform"),
}

#: Stages run when none are given: a translation to get close, then a rigid
#: refinement.
DEFAULT_TRANSFORM_TYPES = ("translation", "rigid")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class ITKElastixBackend:
    """elastix through `itk-elastix`, a native extension.

    One of the two backends :func:`registration_ITKElastix` can run on; the
    other is :class:`multiview_stitcher.browser.elastix.ITKWasmElastixBackend`.
    A backend is asked for three things only, and nothing about the
    registration itself.
    """

    name = "itk"

    install_hint = (
        "Please install the itk-elastix package to use ITKElastix for "
        "registration.\n"
        "E.g. using pip:\n"
        "- `pip install multiview-stitcher[itk-elastix]` or\n"
        "- `pip install itk-elastix`"
    )

    @classmethod
    def load(cls):
        """The `itk` module, or an ImportError saying how to get it."""
        try:
            import itk
        except ImportError:
            raise ImportError(cls.install_hint) from None

        return itk

    def image(self, data, origin, spacing, dims):
        itk = self.load()

        image = itk.image_view_from_array(np.asarray(data, dtype=np.float32))
        # ITK counts its axes the other way round from ours.
        image.SetOrigin(tuple(float(origin[dim]) for dim in dims)[::-1])
        image.SetSpacing(tuple(float(spacing[dim]) for dim in dims)[::-1])
        return image

    def default_parameter_map(self, name, number_of_resolutions):
        itk = self.load()

        return dict(
            itk.ParameterObject.GetDefaultParameterMap(
                name, number_of_resolutions
            )
        )

    def run(
        self,
        parameter_maps,
        fixed,
        moving,
        initial_parameter_object,
        **kwargs,
    ):
        """One elastix call: the resampled moving image and the new chain."""
        itk = self.load()

        initial = itk.ParameterObject.New()
        for parameter_map in initial_parameter_object:
            initial.AddParameterMap(parameter_map)

        result, parameters = itk.elastix_registration_method(
            fixed_image=fixed,
            moving_image=moving,
            parameter_object=parameter_maps,
            initial_transform_parameter_object=initial,
            **{"log_to_console": False, **kwargs},
        )

        return itk.array_view_from_image(result), [
            dict(parameters.GetParameterMap(index))
            for index in range(parameters.GetNumberOfParameterMaps())
        ]


def get_backend(name=None):
    """The elastix backend to register with.

    Without a name, `itk-elastix` is used where it can be - it is what the
    desktop package installs - and the WebAssembly build otherwise, which is
    the only one that exists in the browser.
    """
    if name == "itk":
        return ITKElastixBackend()
    if name == "itkwasm":
        # Imported here rather than at the top of the module: it is the
        # backend that is optional, not the browser, and this keeps a desktop
        # registration from pulling the browser runtime in behind it.
        from multiview_stitcher.browser.elastix import ITKWasmElastixBackend

        return ITKWasmElastixBackend()
    if name is not None:
        raise ValueError(
            f"Unknown elastix backend '{name}'. Available: 'itk', 'itkwasm'."
        )

    try:
        ITKElastixBackend.load()
    except ImportError as itk_missing:
        backend = get_backend("itkwasm")
        try:
            backend.load()
        except ImportError:
            # Neither is installed, so say so about both rather than about
            # whichever happened to be tried last.
            raise ImportError(
                f"{itk_missing}\n\n{backend.install_hint}"
            ) from None
        return backend

    return ITKElastixBackend()


# ---------------------------------------------------------------------------
# Parameter maps
# ---------------------------------------------------------------------------


def initial_transform_parameter_map(initial_affine, ndim):
    """An affine in our axis order as the elastix transform to start from.

    A parameter map rather than a transform file or an in-memory ITK
    transform: it is the one description of a transform both backends read,
    and it comes back unchanged at the head of the chain elastix returns.
    """
    affine = np.asarray(initial_affine, dtype=float)
    itk_matrix = affine[:ndim, :ndim][::-1, ::-1]
    center_of_rotation = np.zeros(ndim, dtype=float)
    itk_offset = (
        affine[:ndim, ndim]
        + (affine[:ndim, :ndim] - np.eye(ndim)) @ center_of_rotation
    )[::-1]

    return {
        "Transform": ["AffineTransform"],
        "NumberOfParameters": [str(ndim * (ndim + 1))],
        "TransformParameters": [
            str(value)
            for value in np.concatenate([itk_matrix.reshape(-1), itk_offset])
        ],
        "CenterOfRotationPoint": [
            str(value) for value in center_of_rotation[::-1]
        ],
        "InitialTransformParameterFileName": ["NoInitialTransform"],
        "HowToCombineTransforms": ["Compose"],
        "FixedImageDimension": [str(ndim)],
        "MovingImageDimension": [str(ndim)],
        "FixedInternalImagePixelType": ["float"],
        "MovingInternalImagePixelType": ["float"],
        "Size": ["1"] * ndim,
        "Index": ["0"] * ndim,
        "Spacing": ["1"] * ndim,
        "Origin": ["0"] * ndim,
        "Direction": [str(value) for value in np.eye(ndim).reshape(-1)],
        "UseDirectionCosines": ["true"],
        "ResampleInterpolator": ["FinalBSplineInterpolator"],
        "Resampler": ["DefaultResampler"],
        "DefaultPixelValue": ["0"],
        "CompressResultImage": ["false"],
        "FinalBSplineInterpolationOrder": ["3"],
        "ResultImagePixelType": ["float32"],
        "ResultImageFormat": ["nii"],
    }


def stage_parameter_map(
    backend,
    transform_type,
    number_of_resolutions=2,
    number_of_iterations=None,
    metric=None,
    write_result_image=False,
):
    """The elastix parameter map for one stage: its defaults, then our edits."""
    default_map, elastix_transform = TRANSFORM_TYPES[transform_type]
    parameter_map = backend.default_parameter_map(
        default_map, number_of_resolutions
    )

    parameter_map["Transform"] = [elastix_transform]
    # The images arrive already placed by the initial transform; letting
    # elastix guess an initialisation of its own would discard that.
    parameter_map["AutomaticTransformInitialization"] = ["false"]
    # Only the last stage's resampled image is used, for the link quality.
    parameter_map["WriteResultImage"] = [str(bool(write_result_image)).lower()]

    if number_of_iterations is not None:
        parameter_map["MaximumNumberOfIterations"] = [
            str(number_of_iterations)
        ] * number_of_resolutions

    if metric is not None:
        parameter_map["Metric"] = [metric]

    return parameter_map


# ---------------------------------------------------------------------------
# Reading a transform back out
# ---------------------------------------------------------------------------


def _axis_flip(ndim):
    """Change of basis between ITK's (x, y, z) and our (z, y, x) axis order."""
    flip = np.eye(ndim + 1)
    flip[:ndim, :ndim] = np.eye(ndim)[::-1]
    return flip


def _rotation_2d(angle):
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin], [sin, cos]])


def _rotation_3d(angle_x, angle_y, angle_z, compute_zyx=False):
    def rotation(axis, angle):
        cos, sin = np.cos(angle), np.sin(angle)
        first, second = [(1, 2), (2, 0), (0, 1)][axis]
        matrix = np.eye(3)
        matrix[first, first] = matrix[second, second] = cos
        matrix[first, second], matrix[second, first] = -sin, sin
        return matrix

    rotations = [
        rotation(axis, angle)
        for axis, angle in enumerate((angle_x, angle_y, angle_z))
    ]
    # ITK composes the three the other way round unless it is told otherwise.
    if compute_zyx:
        return rotations[0] @ rotations[1] @ rotations[2]
    return rotations[2] @ rotations[1] @ rotations[0]


def _versor_rotation(versor):
    """The rotation of an ITK versor, i.e. a unit quaternion's vector part."""
    x, y, z = versor
    w = np.sqrt(max(0.0, 1.0 - (x * x + y * y + z * z)))
    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
            ],
            [
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
            ],
            [
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ],
        ]
    )


def _matrix_and_offset(parameter_map, ndim):
    """One elastix transform as a homogeneous matrix, in ITK axis order.

    Every transform packs its parameters differently and all of them turn
    about a centre of rotation: ``y = M (x - c) + c + t``.
    """
    name = parameter_map["Transform"][0]
    parameters = np.array(parameter_map["TransformParameters"], dtype=float)
    center = np.array(
        parameter_map.get("CenterOfRotationPoint", ["0"] * ndim), dtype=float
    )

    if name == "TranslationTransform":
        matrix, offset = np.eye(ndim), parameters[:ndim]
    elif name == "EulerTransform" and ndim == 2:
        matrix, offset = _rotation_2d(parameters[0]), parameters[1:3]
    elif name == "EulerTransform":
        matrix = _rotation_3d(
            *parameters[:3],
            compute_zyx=parameter_map.get("ComputeZYX", ["false"])[0]
            == "true",
        )
        offset = parameters[3:6]
    elif name == "SimilarityTransform" and ndim == 2:
        matrix = parameters[0] * _rotation_2d(parameters[1])
        offset = parameters[2:4]
    elif name == "SimilarityTransform":
        matrix = parameters[6] * _versor_rotation(parameters[:3])
        offset = parameters[3:6]
    elif name == "AffineTransform":
        matrix = parameters[: ndim**2].reshape(ndim, ndim)
        offset = parameters[ndim**2 : ndim**2 + ndim]
    else:
        raise ValueError(
            f"Cannot convert an elastix '{name}' transform to an affine "
            "matrix."
        )

    homogeneous = np.eye(ndim + 1)
    homogeneous[:ndim, :ndim] = matrix
    homogeneous[:ndim, ndim] = offset + center - matrix @ center
    return homogeneous


def affine_from_parameter_object(parameter_object, ndim):
    """The affine an elastix parameter object amounts to, in our axis order.

    The object is the chain of transforms elastix has accumulated, oldest
    first: the initial transform, then one map per stage that ran. Composing
    means applying them in that order, so the product runs the other way.
    """
    composed = np.eye(ndim + 1)
    for parameter_map in parameter_object:
        composed = _matrix_and_offset(parameter_map, ndim) @ composed

    flip = _axis_flip(ndim)
    return flip @ composed @ flip


# ---------------------------------------------------------------------------
# The registration function
# ---------------------------------------------------------------------------


def registration_ITKElastix(
    fixed_data,
    moving_data,
    *,
    fixed_origin,
    moving_origin,
    fixed_spacing,
    moving_spacing,
    initial_affine,
    transform_types=None,
    backend=None,
    **elastix_registration_kwargs,
):
    """
    Use ITKElastix to perform registration between two spatial images.

    Parameters
    ----------
    transform_types : list of str, optional
        Sequence of transform types to apply in successive stages.
        Supported values: 'Translation', 'Rigid', 'Similarity', 'Affine'.
        By default ['Translation', 'Rigid'].
    backend : str, optional
        Which elastix to run: 'itk' for the `itk-elastix` package, 'itkwasm'
        for the WebAssembly build (`itkwasm-elastix`), which is what the
        browser runtime has. By default whichever is available, preferring
        `itk-elastix`.
    **elastix_registration_kwargs
        Additional keyword arguments. The following are handled explicitly
        and applied to the elastix parameter map for each stage:

        number_of_resolutions : int, optional
            Number of resolution levels in the multi-resolution scheme,
            by default 2.
        number_of_iterations : int, optional
            Maximum number of optimizer iterations per resolution level.
            If None, the elastix default for the chosen transform type is used.
        metric : str, optional
            Similarity metric used by elastix. If None, the elastix default
            for the chosen transform type is used. Common values:

            - 'AdvancedMattesMutualInformation' (default for most transforms)
            - 'AdvancedMeanSquares'
            - 'AdvancedNormalizedCorrelation'
            - 'NormalizedMutualInformation'

        Remaining kwargs are forwarded to the backend's registration call
        (e.g. ``log_to_console=True``, which only the 'itk' backend takes).
    """
    # Imported here rather than at the top of the module: `registration`
    # imports this one for the public name of this function, so the dependency
    # runs the other way only when a registration is actually made.
    from multiview_stitcher.registration import link_quality_metric_func

    # Checked before a backend is loaded: a mistyped stage should not be
    # reported only after an import or a download.
    transform_types = [
        name.lower() for name in (transform_types or DEFAULT_TRANSFORM_TYPES)
    ]
    unsupported = [
        name for name in transform_types if name not in TRANSFORM_TYPES
    ]
    if not transform_types or unsupported:
        raise ValueError(
            "elastix needs at least one transform type"
            + (f", and cannot run {unsupported}" if unsupported else "")
            + f". Available: {sorted(TRANSFORM_TYPES)}."
        )

    backend = get_backend(backend)

    spatial_dims = fixed_data.dims
    ndim = len(spatial_dims)

    fixed_image = backend.image(
        fixed_data.data, fixed_origin, fixed_spacing, spatial_dims
    )
    moving_image = backend.image(
        moving_data.data, moving_origin, moving_spacing, spatial_dims
    )

    number_of_iterations = elastix_registration_kwargs.pop(
        "number_of_iterations", None
    )
    number_of_resolutions = elastix_registration_kwargs.pop(
        "number_of_resolutions", 2
    )
    metric = elastix_registration_kwargs.pop("metric", None)

    # One elastix call per transform type. Each stage is handed the whole
    # chain accumulated so far as the transform to start from - the initial
    # affine, then whatever the stages before it found - and hands back that
    # same chain with its own result appended. This is what elastix does
    # internally for a multi-stage parameter object, done a stage at a time so
    # that the chain is a value we hold rather than a file it writes.
    parameter_object = [initial_transform_parameter_map(initial_affine, ndim)]
    result_image = None

    for index, transform_type in enumerate(transform_types):
        result_image, parameter_object = backend.run(
            [
                stage_parameter_map(
                    backend,
                    transform_type,
                    number_of_resolutions=number_of_resolutions,
                    number_of_iterations=number_of_iterations,
                    metric=metric,
                    write_result_image=index == len(transform_types) - 1,
                )
            ],
            fixed_image,
            moving_image,
            parameter_object,
            **elastix_registration_kwargs,
        )

    return {
        "affine_matrix": param_utils.affine_to_xaffine(
            affine_from_parameter_object(parameter_object, ndim)
        ),
        "quality": link_quality_metric_func(
            np.asarray(fixed_data.data), np.asarray(result_image)
        ),
    }
