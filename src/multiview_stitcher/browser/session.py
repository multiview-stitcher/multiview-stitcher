"""
The stateful object behind the browser UI.

A :class:`Session` owns the opened views, the transform keys registered on
them, and any virtual OME-Zarr the viewer is currently reading. It lives in one
persistent Pyodide worker; compute workers rebuild an equivalent, read-only
copy from :class:`~multiview_stitcher.browser.specs.SessionSpec` whenever they
are handed work, so the same Python code runs everywhere and image data never
crosses a worker boundary.

Cache invalidation is explicit and structural. Every URL a viewer is given
carries a *generation*; anything that changes what those URLs should return
bumps it, which retires the old routes and gives Neuroglancer URLs it has
never seen. Stale requests are answered with "not found" instead of stale
pixels.

Views and derived images are counted separately. A fused preview depends on
the transforms and on the fusion options, so registering retires it. A view
does not: registration reaches the viewer as a Neuroglancer source transform
and changes nothing a view route serves, so those URLs stay put - which is
what lets the viewer re-aim the layers it already has instead of discarding
them, their shaders and their contrast ranges.
"""

import uuid

import numpy as np

from multiview_stitcher import (
    msi_utils,
    mv_graph,
    neuroglancer,
    ngff_utils,
    param_utils,
)
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

#: Route name prefix of the virtual OME-Zarrs exposing input views.
VIEW_PREFIX = "view_"

POSITIONAL_COLOR_PALETTE = [
    "#E69F00",
    "#56B4E9",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#0072B2",
    "#F0E442",
]


class Session:
    """Opened views plus everything derived from them."""

    def __init__(self, session_id=None, fetch=None, write=None):
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.fetch = fetch
        # Writing is only needed when fusing to disk; the browser supplies it
        # through the same service worker that serves reads.
        self.write = write
        self.sources = []
        self.msims = []
        self.generation = 0
        self.views_generation = 0
        # route -> VirtualOMEZarr, valid only for the current generation
        self._virtual_zarrs = {}
        self._preview_options = None

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def load(self, sources, replace=True):
        """Open sources as the session's views.

        With ``replace=False`` the sources are appended to those already open,
        which is what dropping a further tile onto a loaded session does. Only
        the new sources are opened; the existing views keep their transforms.

        The result is validated before anything is committed, so a source that
        does not fit the loaded views raises without leaving the session in a
        half-updated state.
        """
        added = [SourceSpec.from_dict(source) for source in sources]

        if replace:
            sources_after, msims_before = [], []
        else:
            known = {source.url for source in self.sources}
            added = [source for source in added if source.url not in known]
            sources_after, msims_before = list(self.sources), list(self.msims)

        new_msims = browser_dataset.open_msims(added, fetch=self.fetch)

        sources_after = sources_after + added
        msims_after = msims_before + new_msims
        browser_dataset.check_compatible(msims_after)

        self.sources, self.msims = sources_after, msims_after
        self.bump_generation(views=True)
        return self.describe()

    def add(self, sources):
        """Append sources to the ones already open."""
        return self.load(sources, replace=False)

    def remove(self, index):
        """Drop a single view."""
        index = int(index)
        if not 0 <= index < len(self.msims):
            raise IndexError(
                f"View {index} does not exist; the session has "
                f"{len(self.msims)} view(s)."
            )

        del self.sources[index]
        del self.msims[index]
        self.bump_generation(views=True)
        return self.describe()

    def clear(self):
        """Drop every view, returning the session to its empty state."""
        self.sources = []
        self.msims = []
        self.bump_generation(views=True)
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
                | {
                    "url": source.url,
                    "served": (
                        "native"
                        if browser_dataset.is_directly_servable(source)
                        else "virtual"
                    ),
                }
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

    def is_empty(self):
        return not self.msims

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
            views_generation=self.views_generation,
            session_id=self.session_id,
            preview=(
                self._preview_options.to_dict()
                if self._preview_options is not None
                else None
            ),
        )

    @classmethod
    def from_spec(cls, spec, session_id=None, fetch=None, write=None):
        """Rebuild a read-only equivalent of a session in another worker."""
        spec = (
            spec
            if isinstance(spec, SessionSpec)
            else SessionSpec.from_dict(spec)
        )
        if not spec.sources or not spec.session_id:
            # An empty spec would build a session with a fresh random id at
            # generation 0, which then answers "retired generation" for every
            # route it is asked about - a mute 404 in place of a plain bug.
            raise ValueError(
                "Cannot rebuild a session from an empty spec "
                f"(sources: {len(spec.sources)}, "
                f"session_id: {spec.session_id!r})."
            )

        # Routes are derived from the session id and generation, so a rebuilt
        # session must reuse both to answer the viewer's existing URLs.
        session = cls(
            session_id=session_id or spec.session_id, fetch=fetch, write=write
        )
        session.sources = list(spec.sources)
        session.msims = browser_dataset.open_msims(
            session.sources, fetch=fetch
        )
        session.generation = spec.generation
        session.views_generation = (
            spec.generation
            if spec.views_generation is None
            else spec.views_generation
        )

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

    def copy_transform(self, source_transform_key, new_transform_key):
        """Copy a common coordinate system under a new editable name."""
        source_transform_key = (
            source_transform_key or self.default_transform_key()
        )
        new_transform_key = str(new_transform_key or "").strip()
        if not new_transform_key:
            raise ValueError("The new transform key must have a name.")
        if source_transform_key not in self.transform_keys():
            raise ValueError(
                f"Transform key '{source_transform_key}' is not available."
            )
        if new_transform_key in self.transform_keys():
            raise ValueError(
                f"Transform key '{new_transform_key}' already exists."
            )

        params = [
            msi_utils.get_transform_from_msim(msim, source_transform_key).copy(
                deep=True
            )
            for msim in self.msims
        ]
        self.set_params(new_transform_key, params)
        return {
            "source_transform_key": source_transform_key,
            "transform_key": new_transform_key,
            "transform_keys": self.transform_keys(),
            "generation": self.generation,
        }

    def update_neuroglancer_transforms(self, transform_key, updates):
        """Persist source transforms edited in the embedded viewer.

        Neuroglancer expresses translations in output pixels. The session's
        affines express them in physical units, so each spatial row is scaled
        by that dimension's image spacing before it is attached to the msim.
        """
        if transform_key not in self.transform_keys():
            raise ValueError(
                f"Transform key '{transform_key}' is not available."
            )

        params = [
            msi_utils.get_transform_from_msim(msim, transform_key).copy(
                deep=True
            )
            for msim in self.msims
        ]

        for update in updates or []:
            index = int(update["index"])
            if not 0 <= index < len(self.msims):
                raise IndexError(f"View {index} does not exist.")

            spec = update.get("transform") or {}
            rows = np.asarray(spec.get("matrix"), dtype=float)
            if rows.ndim != 2 or rows.shape[1] != rows.shape[0] + 1:
                raise ValueError(
                    f"View {index} has an invalid Neuroglancer transform."
                )

            sim = msi_utils.get_sim_from_msim(self.msims[index])
            sdims = list(si_utils.get_spatial_dims_from_sim(sim))
            source = self.sources[index]
            source_dims = (
                list(sim.dims)
                if not browser_dataset.is_directly_servable(source)
                else list(
                    sim.attrs.get(ngff_utils.NGFF_SOURCE_DIMS_ATTR, sim.dims)
                )
            )
            output_dims = [
                str(dim).rstrip("'^")
                for dim in (spec.get("outputDimensions") or {})
            ]
            if not output_dims:
                output_dims = [
                    "c" if dim == "c'" else dim for dim in source_dims
                ]

            try:
                row_indices = [output_dims.index(dim) for dim in sdims]
                column_indices = [source_dims.index(dim) for dim in sdims]
            except ValueError as exc:
                raise ValueError(
                    f"View {index} transform no longer has the spatial "
                    f"dimensions {sdims}."
                ) from exc

            # A Neuroglancer source transform is not in one set of units: its
            # linear coefficients act on physical coordinates - Neuroglancer
            # rescales them by the dimension scales itself - while only the
            # translation is in output pixels. This is the exact inverse of
            # `_affine_to_neuroglancer_source_transform`, which is what builds
            # the transform the viewer was handed.
            affine = np.eye(len(sdims) + 1)
            affine[:-1, :-1] = rows[np.ix_(row_indices, column_indices)]
            spacing = si_utils.get_spacing_from_sim(sim)
            affine[:-1, -1] = [
                rows[row, -1] * spacing[dim]
                for row, dim in zip(row_indices, sdims)
            ]

            current = params[index]
            t_coords = (
                current.coords["t"].values if "t" in current.dims else None
            )
            params[index] = param_utils.affine_to_xaffine(
                affine, t_coords=t_coords
            )

        self.set_params(transform_key, params)
        return {
            "transform_key": transform_key,
            "transform_keys": self.transform_keys(),
            "generation": self.generation,
        }

    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------

    def bump_generation(self, views=False):
        """Retire the URLs previously handed to the viewer.

        Derived images - the fused preview - are always retired: they are
        computed from the transforms and the options in force when they were
        made. Set ``views`` when the set of views itself changed; a
        registration does not, and reusing those URLs is what lets the viewer
        update in place instead of rebuilding every layer.
        """
        self.generation += 1
        if views:
            self.views_generation = self.generation
        self._virtual_zarrs.clear()
        self._preview_options = None
        return self.generation

    def route_prefix(self):
        return f"{self.session_id}/g{self.generation}"

    def views_route_prefix(self):
        return f"{self.session_id}/g{self.views_generation}"

    def _route(self, name):
        return f"{self.route_prefix()}/{name}.ome.zarr"

    def _is_current(self, route):
        # A view route is judged against the generation of the view set, not
        # against the one derived images use.
        if self._view_index_of(route) is not None:
            return route.startswith(f"{self.views_route_prefix()}/")
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

        # Anything derived from the transforms - a fused preview - is now out
        # of date and its URLs are retired. The views are not: the result
        # reaches the viewer as a Neuroglancer source transform, so not a byte
        # of what a view route serves has changed. Keeping those URLs is what
        # lets the viewer re-aim the layers it already has rather than
        # discarding them, their shaders and their contrast ranges, and
        # refetching data it holds.
        self.bump_generation(views=False)

        return {
            "transform_key": options.new_transform_key,
            "params": serialization.params_to_json(params),
            "transform_keys": self.transform_keys(),
            "generation": self.generation,
        }

    def registration_msims(self, reg_channel=None):
        """Views as `register` prepares them, i.e. reduced to one channel.

        `registration.register` selects the registration channel before
        computing pairwise registrations. Compute workers must apply the same
        selection, otherwise they would register multi-channel stacks and
        return transforms of the wrong rank. The channel is identified by its
        coordinate value rather than an index, so it cannot drift out of step
        with what the caller actually selected.
        """
        if reg_channel is None or not self.msims:
            return self.msims

        return [
            msi_utils.multiscale_sel_coords(msim, {"c": reg_channel})
            if "c" in msi_utils.get_dims(msim)
            else msim
            for msim in self.msims
        ]

    def compute_pairwise(self, edges, register_kwargs, reg_channel=None):
        """Compute a subset of pairwise registrations - the compute-worker side.

        Runs the exact same code path as a local registration; only the set of
        edges differs.
        """
        msims = self.registration_msims(reg_channel)

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

        # A new fusion changes what the preview URL should return, so it gets a
        # new generation - the same rule that retires routes after a
        # registration. Without it the preview would share a generation with
        # the state that preceded it, and a worker that had already rebuilt
        # that state would answer "not found" for every key of an image it has
        # never been told about.
        self.bump_generation()

        return self._build_preview(options)

    def _build_preview(self, options):
        """Construct the preview image for the *current* generation.

        Separate from `fuse_preview` because rebuilding an existing preview -
        which is what a compute worker does on its first chunk request - must
        reproduce the route it was asked for, not mint a new one.
        """
        fused_msim = browser_fusion.preview(self.msims, options)
        route = self._route(PREVIEW_NAME)
        self._virtual_zarrs[route] = ngff_utils.VirtualOMEZarr(
            fused_msim,
            name=f"{PREVIEW_NAME}.ome.zarr",
            omero=browser_fusion.inherited_omero(self.msims),
        )
        self._preview_options = options

        return {
            "route": route,
            "generation": self.generation,
            "metadata": serialization.msim_metadata(
                fused_msim, name=PREVIEW_NAME
            ),
        }

    def positional_colors(self, transform_key=None, n_colors=2):
        """Return one adjacency-based display color for each input view."""
        if self.is_empty():
            return {"colors": []}
        transform_key = transform_key or self.default_transform_key()
        sims = [msi_utils.get_sim_from_msim(msim) for msim in self.msims]
        color_indices = mv_graph.get_greedy_colors(
            sims,
            n_colors=int(n_colors),
            transform_key=transform_key,
        )
        return {
            "colors": [
                POSITIONAL_COLOR_PALETTE[
                    color_indices[index] % len(POSITIONAL_COLOR_PALETTE)
                ]
                for index in range(len(sims))
            ]
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
        """Create every output array and list the blocks each one needs.

        Only array metadata is written here; the pixels are fused afterwards,
        in parallel, by whichever workers are free.
        """
        options = self._fusion_options(options)
        levels = browser_fusion.create_output_arrays(
            self.msims, options, fetch=self.fetch, write=self.write
        )
        return {
            "options": options.to_dict(),
            "levels": levels,
            "n_blocks": sum(len(level["block_ids"]) for level in levels),
        }

    def fuse_blocks(self, options, level, ids):
        """Fuse a subset of one level's blocks - the compute-worker side."""
        options = self._fusion_options(options)
        return browser_fusion.fuse_blocks(
            self.msims,
            options,
            level,
            ids,
            fetch=self.fetch,
            write=self.write,
        )

    def finalize_fusion(self, options):
        """Write the multiscales metadata once every block has been fused."""
        options = self._fusion_options(options)
        written = browser_fusion.write_multiscales_metadata(
            self.msims, options, fetch=self.fetch, write=self.write
        )
        self.bump_generation()
        return {
            "output_zarr_url": options.output_zarr_url,
            "generation": self.generation,
            **written,
        }

    # ------------------------------------------------------------------
    # Serving virtual OME-Zarr to the viewer
    # ------------------------------------------------------------------

    def view_route(self, index):
        """Route of the virtual OME-Zarr exposing input view ``index``."""
        return (
            f"{self.views_route_prefix()}/"
            f"{VIEW_PREFIX}{int(index)}.ome.zarr"
        )

    def _view_index_of(self, route):
        """The view index a route addresses, or None if it is not a view."""
        name = route.rsplit("/", 1)[-1]
        if not name.startswith(VIEW_PREFIX) or not name.endswith(".ome.zarr"):
            return None
        try:
            index = int(name[len(VIEW_PREFIX) : -len(".ome.zarr")])
        except ValueError:
            return None
        return index if 0 <= index < len(self.msims) else None

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

        index = self._view_index_of(route)
        if index is not None:
            # An input view that the viewer cannot read directly, e.g. a
            # generated example. Every worker rebuilds the same image from the
            # session spec, so any of them can answer.
            virtual_zarr = ngff_utils.VirtualOMEZarr(
                self.msims[index], name=route.rsplit("/", 1)[-1]
            )
            self._virtual_zarrs[route] = virtual_zarr
            return virtual_zarr

        if route == self._route(PREVIEW_NAME) and self._preview_options:
            self._build_preview(self._preview_options)
            return self._virtual_zarrs.get(route)

        return None

    def why_missing(self, route):
        """Explain why ``route`` cannot be served, for diagnostics."""
        if self._is_current(route):
            return (
                f"no image is registered at '{route}' in generation "
                f"{self.generation} "
                f"(preview options: {self._preview_options is not None}, "
                f"views: {len(self.msims)})"
            )
        return (
            f"'{route}' belongs to a retired generation; this session is at "
            f"{self.route_prefix()} with {len(self.msims)} view(s)"
        )

    def serve(self, route, key):
        """Answer one virtual OME-Zarr request.

        Returns ``(kind, payload)`` where ``kind`` is 'json', 'bytes' or
        'missing'. For 'missing', the payload is a human-readable reason.
        """
        virtual_zarr = self.ensure_route(route)
        if virtual_zarr is None:
            return "missing", self.why_missing(route)

        key = str(key).strip("/")
        if not key:
            return "missing", "empty key"

        try:
            return "json", virtual_zarr.get_json_key(key)
        except KeyError:
            pass

        try:
            path, chunk_key = virtual_zarr._parse_data_key(key)
            return "bytes", virtual_zarr.read_chunk(path, chunk_key)
        except KeyError:
            return "missing", f"'{key}' is not a key of '{route}'"

    # ------------------------------------------------------------------
    # Viewer state
    # ------------------------------------------------------------------

    def source_url(self, index, origin="", api_base="", serve_views="auto"):
        """The URL the viewer should read view ``index`` from.

        With ``serve_views="auto"`` (the default), OME-Zarr behind the service
        worker is streamed straight to the viewer, and anything else - a
        generated example, or any source that only exists in the Python heap -
        is exposed as a virtual OME-Zarr. ``serve_views="virtual"`` routes every
        view through Python instead, which is slower but works for any input.
        """
        source = self.sources[index]
        native = serve_views != "virtual" and browser_dataset.is_directly_servable(
            source
        )
        if native:
            return f"{origin}{source.url}"
        return f"{origin}{api_base}/zarr/{self.view_route(index)}"

    def neuroglancer_state(
        self,
        transform_key=None,
        base_url="",
        api_base="",
        serve_views="auto",
        include_views=True,
        preview_route=None,
        channel_coord=None,
        contrast_limits=None,
        layout=None,
        show_all_channels=False,
    ):
        """Build the Neuroglancer viewer state for the current session.

        Views carry the selected transform key as a Neuroglancer source
        transform, so switching transform keys never rewrites image data.

        ``api_base`` is the service worker's path prefix. It has to be supplied
        by the page rather than assumed: when the app is published under a
        sub-path, a service worker may only claim URLs inside its own scope,
        so a root-relative guess here would produce URLs nothing intercepts.
        """
        if self.is_empty():
            # Nothing to show yet; return a state the viewer accepts rather
            # than failing, so the page can clear the viewer the same way it
            # updates it.
            return {"layers": [], "layout": "4panel"}

        urls = []
        sims = []

        if include_views:
            urls += [
                "zarr://"
                + self.source_url(
                    index,
                    origin=base_url,
                    api_base=api_base,
                    serve_views=serve_views,
                )
                for index in range(len(self.sources))
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
            # Name the layers as the app lists the views, so the two can be
            # read side by side and a removed view is unambiguous.
            layer_dicts=[
                {"name": f"{index}: {source.resolved_name(index)}"}
                for index, source in enumerate(self.sources)
            ]
            if include_views
            else None,
            source_dims=[
                (
                    tuple(sim.dims)
                    if serve_views == "virtual"
                    or not browser_dataset.is_directly_servable(source)
                    else tuple(
                        sim.attrs.get(
                            ngff_utils.NGFF_SOURCE_DIMS_ATTR, sim.dims
                        )
                    )
                )
                for source, sim in zip(self.sources, sims)
            ]
            if include_views
            else [],
        )

        if show_all_channels and "c" in sims[0].dims:
            channel_coords = [
                str(value) for value in sims[0].coords["c"].values
            ]
            if len(channel_coords) > 1:
                layers = []
                for channel in channel_coords:
                    channel_state = neuroglancer.generate_neuroglancer_json(
                        ome_zarr_paths=None,
                        ome_zarr_urls=urls,
                        sims=sims,
                        transform_key=transform_key,
                        channel_coord=channel,
                        contrast_limits=contrast_limits,
                        layout=layout,
                        layer_dicts=[
                            {
                                "name": (
                                    f"{index}: {source.resolved_name(index)}"
                                    f" · {channel}"
                                )
                            }
                            for index, source in enumerate(self.sources)
                        ],
                        source_dims=[
                            (
                                tuple(sim.dims)
                                if serve_views == "virtual"
                                or not browser_dataset.is_directly_servable(
                                    source
                                )
                                else tuple(
                                    sim.attrs.get(
                                        ngff_utils.NGFF_SOURCE_DIMS_ATTR,
                                        sim.dims,
                                    )
                                )
                            )
                            for source, sim in zip(self.sources, sims)
                        ],
                    )
                    layers.extend(channel_state.get("layers", []))
                state["layers"] = layers

        # Keep Neuroglancer's own layer and shader panels out of the way until
        # the user explicitly opens them from the viewer controls.
        state["layerListPanel"] = {"visible": False}
        state["selectedLayer"] = {"visible": False}

        preview_zarr = (
            self.ensure_route(preview_route) if preview_route else None
        )
        if preview_zarr is not None:
            preview_sim = preview_zarr.sims[0]
            layer = {
                # No "type": an untyped layer is opened as Neuroglancer's
                # "auto" layer, and that is the one it expands into a layer per
                # channel, colouring each from the OME-Zarr's omero metadata.
                # Naming the type here skips that step, which left the fused
                # preview a single grey channel next to coloured input views.
                "source": {
                    "url": (
                        f"zarr://{base_url}{api_base}/zarr/"
                        f"{preview_route}"
                    )
                },
                "tab": "rendering",
                "opacity": 1.0,
                "name": PREVIEW_NAME,
                # A fused image exists only in the coordinate system it was
                # fused in. Shown under a different transform key it would
                # sit somewhere the views are not, so it stays loaded but
                # hidden until that key is selected again.
                "visible": self.preview_matches(transform_key),
            }
            if "c" in preview_sim.dims:
                # This is the same channel-local coordinate setup used for
                # input views. Neuroglancer expands the ready image layer into
                # one managed layer per channel and applies its OMERO color and
                # window metadata to each one.
                layer["localDimensions"] = {"c'": [1, ""]}
                layer["localPosition"] = [0]
            state["layers"] = list(state.get("layers", [])) + [layer]

        return state

    def preview_matches(self, transform_key):
        """Whether the fused preview belongs to ``transform_key``."""
        if self._preview_options is None:
            return True
        if transform_key is None:
            transform_key = self.default_transform_key()
        return self._preview_options.transform_key == transform_key
