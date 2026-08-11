"""
The WebAssembly backend of `elastix.registration_ITKElastix`.

`itk-elastix` is a native extension with no WebAssembly build. The same
elastix compiled to WebAssembly is published as `itkwasm-elastix`, and that
one runs wherever JavaScript or wasmtime does - including inside Pyodide,
where it is what makes anything beyond a translation available to the browser
app.

Only the three things that differ between the two elastix builds live here:
how an image is made, where a transform's defaults come from, and how one
stage is run. The registration itself - the stages, the parameter maps, the
initial transform, reading the affine back out - is
:func:`multiview_stitcher.elastix.registration_ITKElastix`, so this backend
follows any change made there.

Nothing here is imported until a registration asks for elastix. That laziness
is the point: the pipeline is a 16 MB WebAssembly module (2.4 MB over the
wire), fetched on first use and then held for the lifetime of the worker that
ran it, so a session that never selects this method never pays for it.
"""

import functools

import numpy as np

from multiview_stitcher import elastix
from multiview_stitcher.browser import env

_IMPORT_ERROR = (
    "Please install the itkwasm-elastix package to use elastix for "
    "registration in this runtime.\n"
    "E.g. using pip:\n"
    "- `pip install itkwasm-elastix`"
)


class ITKWasmElastixBackend:
    """elastix as WebAssembly, through `itkwasm-elastix`.

    Outside the browser itk-wasm offers a synchronous API. In Pyodide it does
    not: the WebAssembly pipelines are driven from JavaScript, so only the
    ``_async`` half exists there. `run_sync` bridges the two by suspending the
    WebAssembly stack until the promise settles - the same mechanism zarr's
    synchronous API uses in this app, and it works here for the same reason:
    every call into Python is made with `callPromising` (see
    docs/browser/py-runtime.js).
    """

    name = "itkwasm"

    install_hint = _IMPORT_ERROR

    @staticmethod
    def load():
        try:
            import itkwasm_elastix
        except ImportError:
            raise ImportError(_IMPORT_ERROR) from None
        return itkwasm_elastix

    def image(self, data, origin, spacing, dims):
        from itkwasm import FloatTypes, Image, ImageType, PixelTypes

        self.load()
        array = np.ascontiguousarray(data, dtype=np.float32)

        return Image(
            imageType=ImageType(
                dimension=array.ndim,
                componentType=FloatTypes.Float32,
                pixelType=PixelTypes.Scalar,
                components=1,
            ),
            # ITK counts its axes the other way round from ours.
            origin=[float(origin[dim]) for dim in dims][::-1],
            spacing=[float(spacing[dim]) for dim in dims][::-1],
            size=list(array.shape[::-1]),
            data=array,
        )

    def default_parameter_map(self, name, number_of_resolutions):
        module = self.load()

        if not env.is_pyodide():
            return module.default_parameter_map(
                name, number_of_resolutions=number_of_resolutions
            )

        from pyodide.ffi import run_sync

        return run_sync(
            module.default_parameter_map_async(
                name, number_of_resolutions=number_of_resolutions
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
        if kwargs:
            raise TypeError(
                f"The itk-wasm elastix backend takes no {sorted(kwargs)} "
                "keyword(s); they are specific to the itk-elastix package."
            )

        module = self.load()

        if not env.is_pyodide():
            result, _transforms, parameters = module.elastix(
                parameter_maps,
                fixed=fixed,
                moving=moving,
                initial_transform_parameter_object=initial_parameter_object,
            )
            return np.asarray(result.data), parameters

        from itkwasm.pyodide import to_js, to_py
        from itkwasm_elastix_emscripten.js_package import js_package
        from pyodide.ffi import run_sync

        js_module = run_sync(js_package.js_module)
        outputs = run_sync(
            js_module.elastix(
                to_js(parameter_maps),
                fixed=to_js(fixed),
                moving=to_js(moving),
                initialTransformParameterObject=to_js(
                    initial_parameter_object
                ),
                # Run the pipeline in this worker rather than in a web worker
                # of its own: one WebAssembly instance of elastix per Python
                # worker instead of two, and no image copied across a thread
                # boundary.
                webWorker=False,
                noCopy=True,
            )
        )

        # The JavaScript module is called directly, rather than through
        # `itkwasm_elastix.elastix_async`, because that converts *every*
        # output - including the transform list, which itkwasm 1.0b200 cannot
        # convert in Pyodide: the composite header of the list carries the
        # data URI that addressed its (empty) parameter arrays, and the
        # converter reads it as a buffer ("a bytes-like object is required,
        # not 'str'"). Only the two outputs used here are converted, and
        # neither has that problem.
        return (
            np.asarray(to_py(outputs.result).data),
            to_py(outputs.transformParameterObject),
        )


#: What the browser selects as `itk_elastix`: `registration_ITKElastix` with
#: the backend pinned, since which elastix runs must not depend on what else
#: happens to be installed - `itk-elastix` is importable in a development
#: environment and never in Pyodide.
#:
#: A partial rather than a wrapper function: `register` decides what to pass a
#: pairwise registration function by inspecting its signature, and a
#: ``*args, **kwargs`` wrapper has none to inspect.
registration_elastix = functools.partial(
    elastix.registration_ITKElastix, backend="itkwasm"
)
