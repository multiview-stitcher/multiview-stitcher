"""
The stateful object behind the browser UI.

A :class:`Session` owns the opened views, the transform keys registered on
them, and any virtual OME-Zarr the viewer is currently reading. It lives in one
persistent Pyodide worker; compute workers rebuild an equivalent, read-only
copy from :class:`~multiview_stitcher.browser.specs.SessionSpec` whenever they
are handed work, so the same Python code runs everywhere and image data never
crosses a worker boundary.

Cache invalidation is explicit and structural. Every URL a viewer is given
carries the session *generation*; anything that changes what those URLs should
return (new registration results, a new fusion) bumps the generation, which
retires the old routes and gives Neuroglancer URLs it has never seen. Stale
requests are answered with "not found" instead of stale pixels.
"""

import uuid

from multiview_stitcher import msi_utils, neuroglancer, ngff_utils
from multiview_stitcher import registration as core_registration
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.browser import dataset as browser_dataset
from multiview_stitcher.browser import fusion as browser_fusion
from multiview_stitcher.browser import serialization
from multiview_stitcher.browser.specs import (
    FusionOptions,
    RegistrationOptions,
    SessionSpec,
    SourceSpec,
)

#: Route name of the lazily fused preview image.
PREVIEW_NAME = "fused"


class Session:
    """Opened views plus everything derived from them."""

    def __init__(self, session_id=None, fetch=None):
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.fetch = fetch
        self.sources = []
        self.msims = []
        self.generation = 0
        # route -> VirtualOMEZarr, valid only for the current generation
        self._virtual_zarrs = {}
        self._preview_options = None

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def load(self, sources):
        """Open the given OME-Zarr sources as the session's views."""
        self.sources = [SourceSpec.from_dict(source) for source in sources]
        self.msims = browser_dataset.open_msims(
            self.sources, fetch=self.fetch
        )
        browser_dataset.check_compatible(self.msims)
        self.bump_generation()
        return self.describe()

    def describe(self):
        """Metadata for the UI: one entry per view plus session-wide state."""
        return {
            "session_id": self.session_id,
            "generation": self.generation,
            "n_views": len(self.msims),
            "transform_keys": self.transform_keys(),
            "views": [
                serialization.msim_metadata(
                    msim, name=source.resolved_name(index)
                )
                | {"url": source.url}
                for index, (source, msim) in enumerate(
                    zip(self.sources, self.msims)
                )
            ],
        }

    # ------------------------------------------------------------------
    # Transform keys
    # ------------------------------------------------------------------

    def transform_keys(self):
        """Transform keys present on *every* view, in a stable order."""
        if not self.msims:
            return []

        common = set(serialization._transform_keys(self.msims[0]))
        for msim in self.msims[1:]:
            common &= set(serialization._transform_keys(msim))

        # Keep the intrinsic metadata transform first, then the rest sorted.
        default = si_utils.DEFAULT_TRANSFORM_KEY
        ordered = [default] if default in common else []
        ordered += sorted(common - {default})
        return ordered

    def default_transform_key(self):
        """The coordinate system new work starts from.

        The intrinsic OME-Zarr metadata transform when present, otherwise the
        first key the views agree on.
        """
        keys = self.transform_keys()
        if not keys:
            raise RuntimeError("No dataset has been loaded yet.")
        return (
            si_utils.DEFAULT_TRANSFORM_KEY
            if si_utils.DEFAULT_TRANSFORM_KEY in keys
            else keys[0]
        )

    def transforms_json(self):
        """Serialise every common transform key, per view."""
        return {
            transform_key: [
                serialization.transform_from_msim_json(msim, transform_key)
                for msim in self.msims
            ]
            for transform_key in self.transform_keys()
        }

    def spec(self):
        """A snapshot compute workers can rebuild this session from."""
        return SessionSpec(
            sources=list(self.sources),
            transforms=self.transforms_json(),
            generation=self.generation,
            session_id=self.session_id,
            preview=(
                self._preview_options.to_dict()
                if self._preview_options is not None
                else None
            ),
        )

    @classmethod
    def from_spec(cls, spec, session_id=None, fetch=None):
        """Rebuild a read-only equivalent of a session in another worker."""
        spec = (
            spec
            if isinstance(spec, SessionSpec)
            else SessionSpec.from_dict(spec)
        )
        # Routes are derived from the session id and generation, so a rebuilt
        # session must reuse both to answer the viewer's existing URLs.
        session = cls(session_id=session_id or spec.session_id, fetch=fetch)
        session.sources = list(spec.sources)
        session.msims = browser_dataset.open_msims(
            session.sources, fetch=fetch
        )
        session.generation = spec.generation

        for transform_key, params in spec.transforms.items():
            session.set_params(
                transform_key,
                serialization.params_from_json(params),
                bump=False,
            )

        # Remember, but do not build, the preview: the fused image is only
        # constructed if this worker is actually asked for one of its chunks.
        if spec.preview is not None:
            session._preview_options = FusionOptions.from_dict(spec.preview)

        return session

    def set_params(self, transform_key, params, base_transform_key=None,
                   bump=True):
        """Attach one affine per view under ``transform_key``."""
        if len(params) != len(self.msims):
            raise ValueError(
                f"Got {len(params)} transforms for {len(self.msims)} views."
            )

        for msim, param in zip(self.msims, params):
            msi_utils.set_affine_transform(
                msim,
                param,
                transform_key=transform_key,
                base_transform_key=base_transform_key,
            )

        if bump:
            self.bump_generation()

        return transform_key

    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------

    def bump_generation(self):
        """Retire every URL previously handed to the viewer."""
        self.generation += 1
        self._virtual_zarrs.clear()
        self._preview_options = None
        return self.generation

    def route_prefix(self):
        return f"{self.session_id}/g{self.generation}"

    def _route(self, name):
        return f"{self.route_prefix()}/{name}.ome.zarr"

    def _is_current(self, route):
        return route.startswith(f"{self.route_prefix()}/")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, options=None, pairwise_executor=None):
        """Register the views and attach the result as a new transform key."""
        options = (
            options
            if isinstance(options, RegistrationOptions)
            else RegistrationOptions.from_dict(options)
        )

        if (
            options.reg_channel_index is None
            and self.msims
            and "c" in msi_utils.get_dims(self.msims[0])
        ):
            # The browser exposes one "register" button; pick the first channel
            # rather than making the user answer a question they did not ask.
            options.reg_channel_index = 0

        if options.transform_key is None:
            options.transform_key = self.default_transform_key()

        params = core_registration.register(
            self.msims,
            pairwise_executor=pairwise_executor,
            **options.register_kwargs(),
        )

        # register() already wrote the result onto the msims under
        # new_transform_key; the viewer now needs fresh URLs for it.
        self.bump_generation()

        return {
            "transform_key": options.new_transform_key,
            "params": serialization.params_to_json(params),
            "transform_keys": self.transform_keys(),
            "generation": self.generation,
        }

    def registration_msims(self, reg_channel_index=None):
        """Views as `register` prepares them, i.e. reduced to one channel.

        `registration.register` selects the registration channel before
        computing pairwise registrations. Compute workers must apply the same
        selection, otherwise they would register multi-channel stacks and
        return transforms of the wrong rank.
        """
        if reg_channel_index is None or not self.msims:
            return self.msims

        if "c" not in msi_utils.get_dims(self.msims[0]):
            return self.msims

        sim = msi_utils.get_sim_from_msim(self.msims[0])
        reg_channel = sim.coords["c"][int(reg_channel_index)]

        return [
            msi_utils.multiscale_sel_coords(msim, {"c": reg_channel})
            if "c" in msi_utils.get_dims(msim)
            else msim
            for msim in self.msims
        ]

    def compute_pairwise(
        self, edges, register_kwargs, reg_channel_index=None
    ):
        """Compute a subset of pairwise registrations - the compute-worker side.

        Runs the exact same code path as a local registration; only the set of
        edges differs.
        """
        msims = self.registration_msims(reg_channel_index)

        results = []
        for pair in edges:
            index_a, index_b = int(pair[0]), int(pair[1])
            param_ds = core_registration.register_pair_of_msims_over_time(
                msims[index_a],
                msims[index_b],
                **register_kwargs,
            ).compute()
            results.append(serialization.pairwise_result_to_json(param_ds))
        return results

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def fuse_preview(self, options=None):
        """Register a lazily fused image for the viewer to read from.

        Nothing is computed until Neuroglancer asks for a chunk, and every
        chunk request can be answered by any worker.
        """
        options = self._fusion_options(options)
        if not options.is_preview:
            raise ValueError(
                "fuse_preview() requires FusionOptions without an "
                "output_zarr_url."
            )

        fused_msim = browser_fusion.preview(self.msims, options)
        route = self._route(PREVIEW_NAME)
        self._virtual_zarrs[route] = ngff_utils.VirtualOMEZarr(
            fused_msim, name=f"{PREVIEW_NAME}.ome.zarr"
        )
        self._preview_options = options

        return {
            "route": route,
            "generation": self.generation,
            "metadata": serialization.msim_metadata(
                fused_msim, name=PREVIEW_NAME
            ),
        }

    def _fusion_options(self, options):
        if not isinstance(options, FusionOptions):
            options = FusionOptions.from_dict(options)
        if options.transform_key is None:
            # Default to the most recently added coordinate system, which is
            # the registration result once the user has registered.
            keys = self.transform_keys()
            if not keys:
                raise RuntimeError("No dataset has been loaded yet.")
            options.transform_key = keys[-1]
        return options

    def fusion_plan(self, options):
        """Create the output OME-Zarr array and list the blocks to fuse."""
        options = self._fusion_options(options)
        info = browser_fusion.prepare(
            self.msims, options, create_output=True
        )
        return {
            "options": options.to_dict(),
            "nblocks": [int(n) for n in info["nblocks"]],
            "block_ids": browser_fusion.block_ids(info["nblocks"]),
            "output_stack_properties": (
                serialization.stack_properties_to_json(
                    info["output_stack_properties"]
                )
            ),
        }

    def fuse_blocks(self, options, ids):
        """Fuse a subset of blocks - the compute-worker side."""
        options = self._fusion_options(options)
        return browser_fusion.fuse_blocks(self.msims, options, ids)

    def finalize_fusion(self, options, output_stack_properties):
        """Write NGFF metadata and pyramid levels once all blocks are done."""
        options = self._fusion_options(options)
        browser_fusion.finalize(
            self.msims,
            options,
            serialization.stack_properties_from_json(
                output_stack_properties
            ),
        )
        self.bump_generation()
        return {
            "output_zarr_url": options.output_zarr_url,
            "generation": self.generation,
        }

    # ------------------------------------------------------------------
    # Serving virtual OME-Zarr to the viewer
    # ------------------------------------------------------------------

    def ensure_route(self, route):
        """Return the virtual OME-Zarr for ``route``, rebuilding it if needed.

        Compute workers get chunk requests without having run `fuse_preview`
        themselves; as long as the route belongs to the current generation the
        image is reconstructed from the same options, and is therefore
        identical to the one the session worker registered.
        """
        if route in self._virtual_zarrs:
            return self._virtual_zarrs[route]

        if not self._is_current(route):
            # A URL from an earlier generation: the data behind it no longer
            # exists. Reporting "not found" is what stops the viewer from
            # mixing results computed before and after a registration.
            return None

        if route == self._route(PREVIEW_NAME) and self._preview_options:
            self.fuse_preview(self._preview_options)
            return self._virtual_zarrs.get(route)

        return None

    def serve(self, route, key):
        """Answer one virtual OME-Zarr request.

        Returns ``(kind, payload)`` where ``kind`` is 'json', 'bytes' or
        'missing'.
        """
        virtual_zarr = self.ensure_route(route)
        if virtual_zarr is None:
            return "missing", None

        key = str(key).strip("/")
        if not key:
            return "missing", None

        try:
            return "json", virtual_zarr.get_json_key(key)
        except KeyError:
            pass

        try:
            path, chunk_key = virtual_zarr._parse_data_key(key)
            return "bytes", virtual_zarr.read_chunk(path, chunk_key)
        except KeyError:
            return "missing", None

    # ------------------------------------------------------------------
    # Viewer state
    # ------------------------------------------------------------------

    def neuroglancer_state(
        self,
        transform_key=None,
        base_url="",
        include_views=True,
        preview_route=None,
        channel_coord=None,
        contrast_limits=None,
        layout=None,
    ):
        """Build the Neuroglancer viewer state for the current session.

        Input views are served as their native OME-Zarr and carry the selected
        transform key as a Neuroglancer source transform, so switching
        transform keys never rewrites image data.
        """
        urls = []
        sims = []

        if include_views:
            urls += [
                f"zarr://{base_url}{source.url}" for source in self.sources
            ]
            sims += [
                msi_utils.get_sim_from_msim(msim) for msim in self.msims
            ]

        state = neuroglancer.generate_neuroglancer_json(
            ome_zarr_paths=None,
            ome_zarr_urls=urls,
            sims=sims,
            transform_key=transform_key,
            channel_coord=channel_coord,
            contrast_limits=contrast_limits,
            layout=layout,
        )

        if preview_route:
            virtual_zarr = self._virtual_zarrs.get(preview_route)
            if virtual_zarr is not None:
                state["layers"] = list(state.get("layers", [])) + [
                    {
                        "type": "image",
                        "source": {
                            "url": (
                                f"zarr://{base_url}/__mvs__/zarr/"
                                f"{preview_route}"
                            )
                        },
                        "tab": "rendering",
                        "opacity": 1.0,
                        "name": PREVIEW_NAME,
                    }
                ]

        return state
