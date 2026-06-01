import json
import os
import ssl
import urllib
import webbrowser
from functools import partial
from http.server import (
    HTTPServer,
    SimpleHTTPRequestHandler,
    test,
)

import numpy as np
import zarr
from matplotlib import colormaps, colors
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from xarray import DataTree

from multiview_stitcher import (
    msi_utils,
    mv_graph,
    ngff_utils,
    param_utils,
    spatial_image_utils,
)


def plot_positions(
    sims,
    transform_key,
    edges=None,
    edge_color_vals=None,
    edge_linestyles=None,
    edge_linestyle_labels=None,
    edge_cmap=None,
    edge_clims=None,
    edge_label="edge weight",
    use_positional_colors=True,
    n_colors=2,
    nscoord=None,
    display_view_indices=True,
    view_labels=None,
    view_labels_size=10,
    show_plot=True,
    plot_title=None,
    spacing=None,
    output_filename=None,
    points_key=None,
):
    """
    Plot tile / view positions in both 2D or 3D.

    Parameters
    ----------
    sims : list of spatial-image (or multiscale-spatial-image), multiview-stitcher flavor
        The views / tiles to plot
    transform_key : str
        Which transform_key to use for visualization
    use_positional_colors : bool, optional
        This colors the views such that neighboring
        views can be distinguished better (warning: can
        be slow for many views), by default True
    n_colors : int, optional
        How many different colors to use when `use_positional_colors` is True,
        by default 2
    nscoord : dict, optional
        non-spatial coordinate to use for visualization (e.g. {'c': 'EGFP', 't': 0}),
        by default {}
    view_labels : list of str, optional
        Custom labels to use for the views, by default None
    view_labels_size : int, optional
        Size of the view labels, by default 10
    show_plot : bool, optional
        Whether to show the plot, by default True
    edge_linestyles : list or str, optional
        Line styles for edges. If a list/array is provided, it must match the
        number of edges. If a single string is provided, it is applied to all
        edges. By default None, which uses dashed lines for all edges.
    edge_linestyle_labels : dict or list of tuples, optional
        Optional legend mapping for edge line styles. Provide either a dict
        of {linestyle: label} or a list of (linestyle, label) pairs.
    spacing : dict, optional
        Overwrite the sims' spacing for plotting. Useful in the case of images with single
        coordinates for which the spacing is not defined in the metadata.
        By default None
    plot_title : str, optional
        Title of the plot, by default no title
    output_filename : str, optional
        Filename where to save the plot if not None, by default None
    points_key : str, optional
        Name of a point set to overlay. If None, point sets are not plotted.
        By default None.

    Returns
    -------
    fig, ax : matplotlib figure and axis
    """

    if nscoord is None:
        nscoord = {}

    if isinstance(sims[0], DataTree):
        # convert msims to sims for backward compatibility
        sims = [msi_utils.get_sim_from_msim(sim) for sim in sims]
    else:
        # make shallow copy of sim list, which fill be modified further down
        sims = list(sims)

    ndim = spatial_image_utils.get_ndim_from_sim(sims[0])
    sdims = spatial_image_utils.get_spatial_dims_from_sim(sims[0])

    # select a single position for non-spatial dimensions
    for isim, sim in enumerate(sims):
        sdims = spatial_image_utils.get_spatial_dims_from_sim(sim)
        nsdims = [dim for dim in sim.dims if dim not in sdims]
        for nsdim in nsdims:
            # if nsdim in sim.dims and len(sim.coords[nsdim]) > 1:
            if nsdim in sim.dims:
                if nsdim not in nscoord:
                    nscoord[nsdim] = sim.coords[nsdim][0]
                sims[isim] = spatial_image_utils.sim_sel_coords(sim, nscoord)

    if use_positional_colors:
        pos_colors = ["red", "green", "blue", "yellow"]
        greedy_colors = mv_graph.get_greedy_colors(
            sims,
            n_colors=n_colors,
            transform_key=transform_key,
        )
        pos_colors = [
            pos_colors[greedy_colors[iview] % len(pos_colors)]
            for iview in range(len(sims))
        ]

    else:
        pos_colors = ["black"] * len(sims)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_aspect("equal")

    for iview, sim in enumerate(sims):
        sp = spatial_image_utils.get_stack_properties_from_sim(
            sim, transform_key=transform_key
        )

        if spacing is not None:
            sp["spacing"] = spacing

        # if single coordinate, enlargen the domain
        for dim in sdims:
            if sp["shape"][dim] == 1:
                sp["shape"][dim] = 2
                sp["origin"][dim] = sp["origin"][dim] - sp["spacing"][dim] / 2

        plot_stack_props(sp, ax, color=pos_colors[iview])

        if points_key is not None:
            point_positions = _get_point_set_positions_for_plot(
                sim,
                points_key=points_key,
                transform_key=transform_key,
                sdims=sdims,
            )
            if point_positions is not None and len(point_positions):
                ax.scatter(
                    point_positions[:, 0],
                    point_positions[:, 1],
                    point_positions[:, 2],
                    c=pos_colors[iview],
                    edgecolors="k",
                    marker="o",
                    s=20,
                    depthshade=False,
                )

    if display_view_indices:
        for iview, sim in enumerate(sims):
            center = spatial_image_utils.get_center_of_sim(
                sim, transform_key=transform_key
            )
            if ndim == 2:
                y, x = center
                z = 0
            else:
                z, y, x = center
            if view_labels is not None:
                text = view_labels[iview]
            else:
                text = str(iview)
            ax.text(
                z,
                x,
                y,
                text,
                size=view_labels_size,
                zorder=1,
                color="k",
                horizontalalignment="center",
            )

    if edges is not None:
        node_poss = [
            spatial_image_utils.get_center_of_sim(
                sim, transform_key=transform_key
            )
            for sim in sims
        ]
        if ndim == 2:
            node_poss = [[0, p[0], p[1]] for p in node_poss]

        node_poss_mpl = [[p[0], p[2], p[1]] for p in node_poss]

        if edge_color_vals is not None:
            edge_color_vals = np.array(edge_color_vals).astype(float)

            if edge_clims is None:
                edge_clims = [min(edge_color_vals), max(edge_color_vals)]
            norm = colors.Normalize(vmin=edge_clims[0], vmax=edge_clims[1])

            if edge_cmap is None:
                edge_cmap = colormaps.get_cmap(
                    "Spectral",
                )
            elif isinstance(edge_cmap, str):
                edge_cmap = colormaps.get_cmap(edge_cmap)

            edge_cmap.set_bad(color="gray")

            edge_colors = [edge_cmap(norm(val)) for val in edge_color_vals]
        else:
            edge_colors = ["k" for _ in edges]

        if edge_linestyles is None:
            edge_linestyles = ["--"] * len(edges)
        elif isinstance(edge_linestyles, str):
            edge_linestyles = [edge_linestyles] * len(edges)
        elif len(edge_linestyles) != len(edges):
            raise ValueError(
                "edge_linestyles must be None, a string, or match the length of edges."
            )

        # Plot the edges
        for e, color, linestyle in zip(edges, edge_colors, edge_linestyles):
            ax.plot(
                *np.array([node_poss_mpl[e[0]], node_poss_mpl[e[1]]]).T,
                linestyle=linestyle,
                color=color,
            )

        if edge_color_vals is not None:
            sm = plt.cm.ScalarMappable(cmap=edge_cmap)
            sm.set_array(
                list(edge_color_vals) + [edge_clims[0], edge_clims[1]]
            )
            plt.colorbar(sm, label=edge_label, ax=ax)
        if edge_linestyle_labels is not None:
            if isinstance(edge_linestyle_labels, dict):
                label_items = list(edge_linestyle_labels.items())
            elif isinstance(edge_linestyle_labels, (list, tuple)):
                label_items = list(edge_linestyle_labels)
            else:
                raise ValueError(
                    "edge_linestyle_labels must be a dict or a list of tuples."
                )
            used_linestyles = set(edge_linestyles)
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    color="k",
                    linestyle=linestyle,
                    label=label,
                )
                for linestyle, label in label_items
                if linestyle in used_linestyles
            ]
            if legend_handles:
                ax.legend(handles=legend_handles, title="Edge usage", loc="best")

    if ndim == 3:
        ax.set_xlabel("z [μm]")
    ax.set_ylabel("x [μm]")
    ax.set_zlabel("y [μm]")

    ax.set_aspect("equal", adjustable="box")

    if ndim == 3:
        ax.view_init(elev=15, azim=-15, roll=0)
    elif ndim == 2:
        ax.view_init(elev=0, azim=0, roll=0)

    # invert y-axis to match imshow and napari view
    ax.invert_zaxis()

    if plot_title is not None:
        plt.title(plot_title)

    plt.tight_layout()
    if output_filename is not None:
        plt.savefig(output_filename)
    if show_plot:
        plt.show()

    return fig, ax


def _get_point_set_positions_for_plot(sim, points_key, transform_key, sdims):
    if (
        "point_sets" not in sim.attrs
        or points_key not in sim.attrs["point_sets"]
    ):
        return None

    point_set = spatial_image_utils.get_point_set(
        sim,
        points_key=points_key,
    )
    position = point_set["position"]
    for dim in position.dims:
        if dim not in ["point_id", "dim"]:
            position = position.isel({dim: 0})

    position = position.sel(dim=sdims).transpose("point_id", "dim")
    positions = np.asarray(position.values, dtype=float)
    positions = positions[np.all(np.isfinite(positions), axis=1)]
    if not len(positions):
        return np.empty((0, 3))

    affine = spatial_image_utils.get_affine_from_sim(
        sim,
        transform_key=transform_key,
    )
    sel_dict = {
        dim: affine.coords[dim][0].values
        for dim in affine.dims
        if dim not in ["x_in", "x_out"]
    }
    if len(sel_dict):
        affine = affine.sel(sel_dict)
    affine = np.asarray(affine)

    positions_h = np.concatenate(
        [positions, np.ones((len(positions), 1))],
        axis=1,
    )
    positions = (affine @ positions_h.T).T[:, :-1]

    if len(sdims) == 2:
        positions = np.column_stack(
            [np.zeros(len(positions)), positions[:, 1], positions[:, 0]]
        )
    else:
        positions = positions[:, [0, 2, 1]]

    return positions


def _resolve_points_key_for_sim(sim, points_key=None):
    if points_key is None:
        return None

    point_set_keys = list(sim.attrs.get("point_sets", {}).keys())
    if points_key not in point_set_keys:
        raise ValueError(
            f"Point set {points_key!r} not found. "
            f"Available point sets: {point_set_keys}."
        )

    return points_key


def _get_sim_for_imshow(image, resolution_level=0):
    if isinstance(image, DataTree):
        scale_key = f"scale{resolution_level}"
        available_scale_keys = msi_utils.get_sorted_scale_keys(image)
        if scale_key not in available_scale_keys:
            raise ValueError(
                f"Resolution level {resolution_level} not found. "
                f"Available levels: {available_scale_keys}."
            )

        return msi_utils.get_sim_from_msim(image, scale=scale_key)

    if resolution_level != 0:
        raise ValueError(
            "resolution_level is only supported for multiscale inputs."
        )

    return image


def imshow(
    image,
    resolution_level=0,
    points_key=None,
    points_tolerance=1,
    project_dim=None,
    horizontal_dim=None,
    vertical_dim=None,
    figure_kwargs=None,
    imshow_kwargs=None,
    scatter_kwargs=None,
    show_plot=True,
):
    """
    Display a 2D slice or projection from a sim or msim.

    The viewer adds one interactive slider per dimension that is not shown in
    the displayed 2D plane (for example z, t, c). Points are converted from
    physical coordinates to
    pixel coordinates as ``(points - origin) / spacing`` for slice matching,
    while the displayed axes and optional point overlay use physical
    coordinates.

    Parameters
    ----------
    image : xarray.DataArray (spatial-image) or xarray.DataTree (multiscale-spatial-image)
        Spatial image or multiscale spatial image.
    resolution_level : int, optional
        Resolution level to display for multiscale inputs. By default 0.
    points_key : str, optional
        Name of the point set to display. If None, no point set is shown.
    points_tolerance : float, optional
        Pixel-space tolerance around the currently selected index for every
        spatial slider dimension. By default 1.
    project_dim : str, optional
        Spatial dimension to maximum-project before display. By default None.
    horizontal_dim : str, optional
        Spatial dimension to plot on the horizontal axis. If None, defaults to
        ``"x"`` when available, otherwise another non-projected spatial
        dimension.
    vertical_dim : str, optional
        Spatial dimension to plot on the vertical axis. If None, defaults to
        ``"y"`` when available, otherwise another non-projected spatial
        dimension.
    figure_kwargs : dict, optional
        Keyword arguments passed to ``matplotlib.pyplot.figure``.
    imshow_kwargs : dict, optional
        Keyword arguments passed to ``Axes.imshow``.
    scatter_kwargs : dict, optional
        Keyword arguments passed to ``Axes.scatter``.
    show_plot : bool, optional
        Whether to call ``plt.show()``. By default True.

    Returns
    -------
    fig, ax, sliders
        Matplotlib figure, axis, and a dict of sliders keyed by dimension.
    """

    points_tolerance = float(points_tolerance)
    if points_tolerance < 0:
        raise ValueError("points_tolerance must be >= 0.")

    figure_kwargs = {} if figure_kwargs is None else dict(figure_kwargs)
    imshow_kwargs = {} if imshow_kwargs is None else dict(imshow_kwargs)
    scatter_kwargs = {} if scatter_kwargs is None else dict(scatter_kwargs)

    sim = _get_sim_for_imshow(image, resolution_level=resolution_level)
    scale_key = f"scale{resolution_level}"
    points_key = _resolve_points_key_for_sim(sim, points_key=points_key)
    if "y" not in sim.dims or "x" not in sim.dims:
        raise ValueError("The selected image must include both y and x dimensions.")

    sdims = spatial_image_utils.get_spatial_dims_from_sim(sim)
    for dim_name, dim_value in [
        ("horizontal_dim", horizontal_dim),
        ("vertical_dim", vertical_dim),
    ]:
        if dim_value is not None and dim_value not in sdims:
            raise ValueError(
                f"{dim_name} must be one of {sdims}, got {dim_value!r}."
            )

    if project_dim is not None:
        if project_dim not in sdims:
            raise ValueError(
                f"project_dim must be one of {sdims}, got {project_dim!r}."
            )

    available_plot_dims = [dim for dim in sdims if dim != project_dim]

    def _resolve_plot_dim(requested_dim, preferred_dims, dim_name, other_dim=None):
        if requested_dim is not None:
            if requested_dim == project_dim:
                raise ValueError(f"{dim_name} must differ from project_dim.")
            return requested_dim

        for dim in preferred_dims:
            if dim in available_plot_dims and dim != other_dim:
                return dim

        for dim in available_plot_dims:
            if dim != other_dim:
                return dim

        raise ValueError(
            "Plotting requires two displayed spatial dimensions. "
            f"After applying project_dim={project_dim!r}, available dimensions "
            f"are {available_plot_dims}."
        )

    horizontal_dim = _resolve_plot_dim(
        horizontal_dim,
        preferred_dims=["x", "z", "y"],
        dim_name="horizontal_dim",
        other_dim=vertical_dim,
    )
    vertical_dim = _resolve_plot_dim(
        vertical_dim,
        preferred_dims=["y", "z", "x"],
        dim_name="vertical_dim",
        other_dim=horizontal_dim,
    )

    if horizontal_dim == vertical_dim:
        raise ValueError("horizontal_dim and vertical_dim must be different.")

    display_spatial_dims = [vertical_dim, horizontal_dim]

    slice_dims = [
        dim
        for dim in sim.dims
        if dim not in display_spatial_dims and dim != project_dim
    ]
    slider_dims = [dim for dim in slice_dims if sim.sizes[dim] > 1]
    slice_indices = {dim: 0 for dim in slice_dims}

    spatial_slice_dims = [dim for dim in slice_dims if dim in sdims]
    sdim_indices = {dim: index for index, dim in enumerate(sdims)}
    row_dim, col_dim = display_spatial_dims
    row_index = sdim_indices[row_dim]
    col_index = sdim_indices[col_dim]

    point_position = None
    if points_key is not None:
        point_position = spatial_image_utils.get_point_set(
            sim,
            points_key=points_key,
        )["position"].sel(dim=sdims)

    origin_dict = spatial_image_utils.get_origin_from_sim(sim)
    spacing_dict = spatial_image_utils.get_spacing_from_sim(sim)
    shape = spatial_image_utils.get_shape_from_sim(sim)
    origin = np.array([origin_dict[dim] for dim in sdims], dtype=float)
    spacing = np.array([spacing_dict[dim] for dim in sdims], dtype=float)

    extent = [
        origin_dict[col_dim] - spacing_dict[col_dim] / 2,
        origin_dict[col_dim]
        + (shape[col_dim] - 0.5) * spacing_dict[col_dim],
        origin_dict[row_dim]
        + (shape[row_dim] - 0.5) * spacing_dict[row_dim],
        origin_dict[row_dim] - spacing_dict[row_dim] / 2,
    ]

    n_sliders = len(slider_dims)
    bottom_margin = 0.14 + n_sliders * 0.06
    fig = plt.figure(**figure_kwargs)
    ax = fig.add_axes([0.1, bottom_margin, 0.8, max(0.2, 0.82 - bottom_margin)])

    def _get_current_image_slice():
        sim_slice = sim.isel(slice_indices) if len(slice_indices) else sim
        if project_dim is not None:
            sim_slice = sim_slice.max(dim=project_dim)
        sim_slice = sim_slice.transpose(*display_spatial_dims)
        return np.asarray(sim_slice.data)

    image_defaults = {
        "interpolation": "nearest",
        "cmap": "gray",
        "extent": extent,
        "origin": "upper",
    }
    image_defaults.update(imshow_kwargs)
    image_artist = ax.imshow(
        _get_current_image_slice(),
        **image_defaults,
    )
    points_artist = None
    if points_key is not None:
        scatter_defaults = {
            "s": 40,
            "edgecolor": "red",
            "facecolor": "none",
        }
        scatter_defaults.update(scatter_kwargs)
        points_artist = ax.scatter(
            [],
            [],
            **scatter_defaults,
        )

    ax.set_xlabel(col_dim)
    ax.set_ylabel(row_dim)
    ax.set_aspect("equal")

    def _update_title():
        title_parts = [scale_key]
        if points_key is not None:
            title_parts.append(f"points={points_key}")
        if project_dim is not None:
            title_parts.append(f"project={project_dim}")
        if len(slider_dims):
            coord_text = ", ".join(
                [
                    f"{dim}={sim.coords[dim].values[slice_indices[dim]]}"
                    for dim in slider_dims
                ]
            )
            title_parts.append(coord_text)
        ax.set_title(" | ".join(title_parts))

    def _update_points_overlay():
        if points_artist is None or point_position is None:
            return

        selected_points = point_position
        for dim in slice_dims:
            if dim in selected_points.dims and dim not in ["point_id", "dim"]:
                selected_points = selected_points.isel({dim: slice_indices[dim]})

        points = np.asarray(selected_points.values, dtype=float)
        if points.ndim == 1:
            points = points[None, :]
        points = points.reshape(-1, len(sdims))

        points = points[np.all(np.isfinite(points), axis=1)]
        if not len(points):
            points_artist.set_offsets(np.empty((0, 2)))
            return

        points_px = (points - origin) / spacing

        keep = np.ones(len(points_px), dtype=bool)
        for dim in spatial_slice_dims:
            dim_index = sdim_indices[dim]
            keep &= (
                np.abs(points_px[:, dim_index] - slice_indices[dim])
                <= points_tolerance
            )

        points_xy = points[keep][:, [col_index, row_index]]
        points_artist.set_offsets(points_xy if len(points_xy) else np.empty((0, 2)))

    sliders = {}

    def _update(_=None):
        for dim, slider in sliders.items():
            slice_indices[dim] = int(slider.val)
            slider.valtext.set_text(str(sim.coords[dim].values[slice_indices[dim]]))

        image_artist.set_data(_get_current_image_slice())
        if points_artist is not None:
            _update_points_overlay()
        _update_title()
        fig.canvas.draw_idle()

    for idim, dim in enumerate(slider_dims):
        slider_ax = fig.add_axes(
            [0.15, 0.03 + (n_sliders - idim - 1) * 0.05, 0.75, 0.03]
        )
        sliders[dim] = Slider(
            ax=slider_ax,
            label=dim,
            valmin=0,
            valmax=sim.sizes[dim] - 1,
            valinit=0,
            valstep=1,
        )
        sliders[dim].on_changed(_update)

    _update()

    if show_plot:
        plt.show()

    return fig, ax, sliders


def plot_msim_with_points(*args, **kwargs):
    return imshow(*args, **kwargs)


def plot_stack_props(stack_props, ax, color="black", size=10, linewidth=1):
    ndim = mv_graph.get_ndim_from_stack_props(stack_props)
    vertices = mv_graph.get_vertices_from_stack_props(stack_props)

    # Build box edges by topology (index hypercube), then map to transformed
    # vertex coordinates. This is robust for arbitrary affine transforms.
    gv = np.array(list(np.ndindex(tuple([2] * ndim))))
    edge_pairs = [
        (i, j)
        for i in range(len(gv))
        for j in range(i + 1, len(gv))
        if np.sum(np.abs(gv[i] - gv[j])) == 1
    ]
    line_segments = np.array([[vertices[i], vertices[j]] for i, j in edge_pairs])

    if ndim == 2:
        line_segments = np.concatenate(
            [np.zeros((line_segments.shape[0], 2, 1)), line_segments], axis=-1
        )

    line_collection = Line3DCollection(
        line_segments[:, :, [0, 2, 1]], colors=color, linewidths=linewidth
    )

    ax.add_collection3d(line_collection)


def plot_tile_pair_image_metrics(
    msims,
    reg_metrics_result,
    base_transform_key,
    query_transform_keys,
    metric_key=None,
    clims=None,
    show_bboxes=True,
    show_overview_plot=False,
    overview_pair_linewidth=1.0,
    show_plot_positions=True,
):
    """
    Visualise registration quality metrics for each query transform key.

    For every entry in *query_transform_keys* a separate figure is produced.
    Each figure shows the tile layout **in that query transform key's world
    coordinate space** and overlays either the pairwise comparison bounding
    boxes (when *show_bboxes* is ``True``) or a minimalistic graph where edges
    are coloured by the metric value (when *show_bboxes* is ``False``).

    The comparison bboxes, which are originally defined in *base_transform_key*
    world space, are projected into each query key's world space via
    ``T_fixed_q @ inv(T_fixed_base)`` (applied to the fixed tile of each pair)
    before being drawn.

    All figures share the same colorbar limits, derived by default from the
    *base_transform_key* metric values when it is included as a query key,
    so all other query keys are compared against the same reference scale.

    Parameters
    ----------
    msims : list of MultiscaleSpatialImage
        The input views, passed unchanged to :func:`plot_positions`.
    reg_metrics_result : dict
        The dictionary returned by :func:`multiview_stitcher.metrics.tile_pair_image_metrics`.
        Must contain the ``"pairs"`` and ``"bboxes"`` keys.
    base_transform_key : str
        Transform key used to define the original comparison bboxes and to set
        colorbar limits when it appears in *query_transform_keys*.
    query_transform_keys : str or list of str
        Subset of transform keys to visualise.  Each key must appear in
        *reg_metrics_result["pairs"]*.  Tile positions and comparison bboxes
        are shown in each key's own world coordinate space.
    metric_key : str, optional
        Name of the metric to use for colouring the comparison boxes or edges.
        Defaults to the first metric key found in the result.
    clims : tuple of (float, float), optional
        Explicit ``(vmin, vmax)`` for the shared colorbar.  When ``None``
        (default) the limits are computed from *base_transform_key* values
        if that key is present in the result, falling back to all query-key
        values otherwise.
    show_bboxes : bool, optional
        When ``True`` (default) the comparison bounding boxes are drawn and
        coloured by metric value.  When ``False`` a minimalistic
        :func:`plot_positions` plot is produced instead, where edges between
        adjacent tiles are coloured by the (mean of the two directed)
        metric values.
    show_overview_plot : bool, optional
        When ``True``, produce one additional figure showing a paired plot
        with *query_transform_keys* on the x-axis and the metric value on the
        y-axis for each pair.  A mean ± std summary (black diamond + error
        bar) is overlaid for each transform key.  By default ``False``.
    overview_pair_linewidth : float, optional
        Line width for the per-pair lines in the overview plot.  Set to
        ``0`` to suppress the lines entirely and show only the mean ± std
        summary markers.  By default ``1.0``.
    show_plot_positions : bool, optional
        When ``True`` (default) the per-query-key positional plots (tile
        layout with coloured comparison bboxes or coloured edges) are
        produced.  Set to ``False`` to skip them, e.g. when only the
        overview plot is needed.

    Returns
    -------
    dict[str, tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]]
        Maps each query transform key to its ``(fig, ax)`` pair.
    """
    if isinstance(query_transform_keys, str):
        query_transform_keys = [query_transform_keys]

    pairs_dict = reg_metrics_result["pairs"]
    bboxes_dict = reg_metrics_result.get("bboxes", {})

    sims = [msi_utils.get_sim_from_msim(msim) for msim in msims]
    spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(sims[0])

    # Collect all available metric keys from the result
    available_metric_keys = set()
    for q_metrics in pairs_dict.values():
        for m in q_metrics.values():
            available_metric_keys.update(m.keys())

    # Determine which metric to colour by
    if metric_key is None:
        metric_key = sorted(available_metric_keys)[0] if available_metric_keys else None
    elif metric_key not in available_metric_keys:
        raise ValueError(
            f"metric_key {metric_key!r} not found in metrics result. "
            f"Available metric keys: {sorted(available_metric_keys)}"
        )

    # Resolve colorbar limits
    if clims is not None:
        vmin, vmax = float(clims[0]), float(clims[1])
    else:
        ref_keys = (
            [base_transform_key]
            if base_transform_key in query_transform_keys
            else query_transform_keys
        )
        ref_values = []
        for pair_metrics in pairs_dict.values():
            for q in ref_keys:
                val = pair_metrics.get(q, {}).get(metric_key, np.nan)
                try:
                    val_f = float(val)
                except (TypeError, ValueError):
                    val_f = np.nan
                if not np.isnan(val_f):
                    ref_values.append(val_f)

        if len(ref_values) >= 2 and min(ref_values) < max(ref_values):
            vmin, vmax = min(ref_values), max(ref_values)
        elif ref_values:
            vmin = ref_values[0] - 0.5
            vmax = ref_values[0] + 0.5
        else:
            vmin, vmax = 0.0, 1.0

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = colormaps.get_cmap("Spectral")

    # Build the list of undirected edges (averaged over both directions) once,
    # used only when show_bboxes=False.
    if not show_bboxes:
        seen = {}
        for (fi, mi) in pairs_dict:
            key = tuple(sorted((fi, mi)))
            if key not in seen:
                seen[key] = []
            seen[key].append((fi, mi))
        undirected_edges = list(seen.keys())

    plots = {}
    for q in query_transform_keys:
        if not show_plot_positions:
            continue
        if show_bboxes:
            fig, ax = plot_positions(
                msims,
                transform_key=q,
                use_positional_colors=False,
                show_plot=False,
                plot_title=f"{metric_key}  |  transform key: {q}",
            )

            for (fi, mi), bbox in bboxes_dict.items():
                if bbox is None:
                    continue

                val = pairs_dict.get((fi, mi), {}).get(q, {}).get(metric_key, np.nan)
                try:
                    val_f = float(val)
                except (TypeError, ValueError):
                    val_f = np.nan

                color = (
                    cmap(norm(val_f)) if not np.isnan(val_f) else (0.5, 0.5, 0.5, 1.0)
                )

                lower = bbox["lower"]
                upper = bbox["upper"]

                # Project the bbox from base_transform_key world space into
                # query key world space using the fixed tile's transforms:
                # T_fixed_q @ inv(T_fixed_base) maps a base-world point to
                # query-world, so bbox corners are visualised at the correct
                # location for each query key.
                T_fixed_base = (
                    spatial_image_utils.get_affine_from_sim(sims[fi], base_transform_key)
                    .squeeze()
                    .data
                )
                T_fixed_q = (
                    spatial_image_utils.get_affine_from_sim(sims[fi], q)
                    .squeeze()
                    .data
                )
                bbox_transform = T_fixed_q# @ np.linalg.inv(T_fixed_base)

                sp = {
                    "origin": {
                        dim: float(lower[idim])
                        for idim, dim in enumerate(spatial_dims)
                    },
                    "spacing": {
                        dim: float(upper[idim] - lower[idim])
                        for idim, dim in enumerate(spatial_dims)
                    },
                    "shape": {dim: 2 for dim in spatial_dims},
                    "transform": param_utils.affine_to_xaffine(bbox_transform),
                }
                plot_stack_props(sp, ax, color=color, linewidth=2)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label=metric_key)

        else:
            # Minimalistic mode: colour edges by mean metric value over both
            # directed pairs.
            edge_color_vals = []
            for fi, mi in undirected_edges:
                directed = seen[(fi, mi)]
                vals = [
                    float(pairs_dict.get(d, {}).get(q, {}).get(metric_key, np.nan))
                    for d in directed
                ]
                valid = [v for v in vals if not np.isnan(v)]
                edge_color_vals.append(float(np.mean(valid)) if valid else np.nan)

            fig, ax = plot_positions(
                msims,
                transform_key=q,
                use_positional_colors=False,
                edges=undirected_edges,
                edge_color_vals=edge_color_vals,
                edge_cmap=cmap,
                edge_clims=[vmin, vmax],
                edge_label=metric_key,
                show_plot=False,
                plot_title=f"{metric_key}  |  transform key: {q}",
            )

        plt.show()

        plots[q] = (fig, ax)

    # ------------------------------------------------------------------
    # Overview plots: one figure per metric key
    # ------------------------------------------------------------------
    if show_overview_plot:
        n_keys = len(query_transform_keys)
        x_positions = list(range(n_keys))

        for mk in [metric_key]:
            fig_ov, ax_ov = plt.subplots(figsize=(max(3.5, 1.6 * n_keys + 1.2), 3.8))

            # Collect per-pair values across query keys
            pair_keys = list(pairs_dict.keys())
            all_vals_flat = []
            pair_series = []
            for pair in pair_keys:
                y_vals = []
                for q in query_transform_keys:
                    raw = pairs_dict[pair].get(q, {}).get(mk, np.nan)
                    try:
                        y_vals.append(float(raw))
                    except (TypeError, ValueError):
                        y_vals.append(np.nan)
                pair_series.append(y_vals)
                all_vals_flat.extend([v for v in y_vals if not np.isnan(v)])

            # Per-pair lines
            if overview_pair_linewidth > 0:
                for y_vals in pair_series:
                    if any(not np.isnan(v) for v in y_vals):
                        ax_ov.plot(
                            x_positions,
                            y_vals,
                            color="#9e9e9e",
                            alpha=0.55,
                            linewidth=overview_pair_linewidth,
                            marker="o",
                            markersize=3.5,
                            zorder=2,
                        )

            # Mean ± std summary per transform key
            means, stds = [], []
            for ix, q in enumerate(query_transform_keys):
                vals = [
                    float(pairs_dict[pair].get(q, {}).get(mk, np.nan))
                    for pair in pair_keys
                ]
                vals = [v for v in vals if not np.isnan(v)]
                if vals:
                    mean_v = float(np.mean(vals))
                    std_v = float(np.std(vals))
                    means.append(mean_v)
                    stds.append(std_v)
                    ax_ov.errorbar(
                        ix,
                        mean_v,
                        yerr=std_v,
                        fmt="o",
                        color="#1f77b4",
                        markersize=8,
                        linewidth=2,
                        capsize=5,
                        capthick=2,
                        zorder=4,
                    )

            # Connect the mean points with a line for easy trend reading
            valid_x = [ix for ix, q in enumerate(query_transform_keys)
                       if any(not np.isnan(float(pairs_dict[pair].get(q, {}).get(mk, np.nan)))
                              for pair in pair_keys)]
            if len(valid_x) > 1:
                mean_y = []
                for ix in valid_x:
                    q = query_transform_keys[ix]
                    vals = [float(pairs_dict[pair].get(q, {}).get(mk, np.nan))
                            for pair in pair_keys]
                    vals = [v for v in vals if not np.isnan(v)]
                    mean_y.append(float(np.mean(vals)) if vals else np.nan)
                ax_ov.plot(valid_x, mean_y, color="#1f77b4", linewidth=1.5,
                           zorder=3, alpha=0.8)

            ax_ov.set_xticks(x_positions)
            ax_ov.set_xticklabels(query_transform_keys, rotation=20, ha="right",
                                  fontsize=10)
            ax_ov.set_ylabel(mk, fontsize=11)
            ax_ov.set_xlim(-0.5, n_keys - 0.5)
            ax_ov.spines["top"].set_visible(False)
            ax_ov.spines["right"].set_visible(False)
            ax_ov.tick_params(axis="both", labelsize=9)
            ax_ov.grid(axis="y", color="#e0e0e0", linewidth=0.8, zorder=0)
            plt.tight_layout()

            plt.show()

    return plots


def serve_dir(dir_path, port=8000):
    """
    Serve a directory with a simple HTTP server.

    Parameters
    ----------
    dir_path : str
        Path to the directory to serve
    port : int, optional
        Port to use for the server, by default 8000
    """

    # code taken from ome-zarr-py
    class CORSRequestHandler(SimpleHTTPRequestHandler):
        def end_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            SimpleHTTPRequestHandler.end_headers(self)

        def translate_path(self, path: str) -> str:
            # Since we don't call the class constructor ourselves,
            # we set the directory here instead
            self.directory = dir_path
            super_path = super().translate_path(path)
            return super_path

    print("Serving images until interrupted...")

    # start serving content
    # suppress console output?
    test(CORSRequestHandler, HTTPServer, port=port)


def serve_dir_https(dir_path, port=8000, host="0.0.0.0", certfile="cert.pem", keyfile="key.pem", quiet=False):
    """
    Serve a directory over HTTPS with a simple HTTP server.

    Before serving, create a self-signed certificate (if not already available):

    ```bash
    # Linux / macOS (needs OpenSSL)
    HOSTNAME="$(hostname -f)"                    # or put your DNS name
    IP="10.0.12.34"                              # your machine's LAN IP

    cat > san.cnf <<EOF
    subjectAltName=DNS:${HOSTNAME},IP:${IP}
    EOF

    openssl req -x509 -newkey rsa:2048 -nodes -days 30 \
    -keyout key.pem -out cert.pem \
    -subj "/CN=${HOSTNAME}" \
    -addext "$(cat san.cnf)"
    ```

    Parameters
    ----------
    dir_path : str
        Path to the directory to serve
    port : int, optional
        Port to use for the server (default 8443)
    host : str, optional
        Host/IP to bind (default "0.0.0.0" = all interfaces)
    certfile : str, optional
        Path to the TLS certificate (PEM)
    keyfile : str, optional
        Path to the TLS private key (PEM)
    quiet : bool, optional
        Suppress request logs if True
    """

    class CORSRequestHandler(SimpleHTTPRequestHandler):
        def end_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            # (optional) add more CORS if you need:
            # self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            # self.send_header("Access-Control-Allow-Headers", "*")
            super().end_headers()

        def translate_path(self, path: str) -> str:
            # set directory dynamically (like your original)
            self.directory = dir_path
            return super().translate_path(path)

        if quiet:
            def log_message(self, fmt, *args):  # noqa: N802
                pass

    handler = partial(CORSRequestHandler, directory=dir_path)
    httpd = HTTPServer((host, port), handler)

    # Wrap the socket with TLS
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=certfile, keyfile=keyfile)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    print(f"Serving {dir_path} over HTTPS at https://{host}:{port} (Ctrl+C to stop)")
    httpd.serve_forever()


def get_contrast_min_max_from_ome_zarr_omero_metadata(
    ome_zarr_path, channel_label=None
):
    """
    Get contrast limits from the OME-Zarr omero metadata key
    for a specific channel. If channel_label is None, the
    first channel is used.
    """
    root = zarr.open_group(ome_zarr_path, mode="r")

    if "omero" not in root.attrs:
        return None

    omero = root.attrs["omero"]

    if channel_label is None:
        channel_index = 0
    else:
        channel_matches = [
            ic
            for ic, c in enumerate(omero["channels"])
            if str(c["label"]) == str(channel_label)
        ]

        if not len(channel_matches) == 1:
            raise ValueError(
                f"Channel {channel_label} not found in metadata in {ome_zarr_path}"
            )
        else:
            channel_index = channel_matches[0]

    window = omero["channels"][channel_index]["window"]

    return np.array([window["start"], window["end"]])


def _affine_to_neuroglancer_source_transform(
    affine, sdims, output_spacing
):
    """
    Convert a physical-space affine to a Neuroglancer source transform.

    OME-Zarr scale and translation already map pixel coordinates into the
    source coordinate space. Neuroglancer rescales source-transform linear
    coefficients from input to output dimension scales internally, but its
    translation coefficients are expressed directly in output coordinate units.
    """
    affine = np.array(affine, dtype=float, copy=True)
    affine_ndim = affine.shape[-1] - 1
    affine_sdims = sdims[-affine_ndim:]
    output_spacing_array = np.array(
        [output_spacing[dim] for dim in affine_sdims]
    )
    affine[:-1, -1] = affine[:-1, -1] / output_spacing_array
    return affine


def generate_neuroglancer_json(
    ome_zarr_paths: list[str],
    ome_zarr_urls: list[str],
    sims: list = None,
    source_images: list = None,
    transform_key: str = None,
    channel_coord: str = None,
    single_layer: bool = False,
    contrast_limits: tuple = None,
    layer_dicts: list[dict] = None,
    global_dict: dict = None,
):
    # read the first multiscales
    if source_images is None:
        sim = ngff_utils.read_sim_from_ome_zarr(ome_zarr_paths[0])
    else:
        sim = source_images[0]
    sdims = spatial_image_utils.get_spatial_dims_from_sim(sim)
    ndim = len(sdims)
    dims = sim.dims
    spacing = spatial_image_utils.get_spacing_from_sim(sim)

    if sims is not None:
        if transform_key is None:
            raise ValueError(
                "transform_key must be provided if sims are given"
            )

        full_affines = [np.eye(len(dims) + 1) for _ in sims]
        spacings_per_sim = []
        for isim, sim in enumerate(sims):

            if source_images is None:
                sim_ome_zarr = ngff_utils.read_sim_from_ome_zarr(
                    ome_zarr_paths[isim]
                )
            else:
                sim_ome_zarr = source_images[isim]
            spacing_zarr = spatial_image_utils.get_spacing_from_sim(sim_ome_zarr)
            spacing_isim = spacing_zarr
            spacings_per_sim.append(spacing_isim)

            affine = spatial_image_utils.get_affine_from_sim(
                sim, transform_key=transform_key
            )
            if "t" in affine.dims:
                affine = affine.sel(t=0)

            # Compose a correction that maps from OME-Zarr physical coordinates to
            # in-memory physical coordinates before applying the registered affine.
            # This is needed when the user has modified origin/spacing of the in-memory
            # sim relative to what is stored in the OME-Zarr on disk.
            affine_np = np.array(affine, dtype=float)
            affine_ndim = affine_np.shape[-1] - 1
            affine_sdims = sdims[-affine_ndim:]
            origin_zarr = spatial_image_utils.get_origin_from_sim(sim_ome_zarr)
            origin_mem = spatial_image_utils.get_origin_from_sim(sim)
            spacing_mem = spatial_image_utils.get_spacing_from_sim(sim)
            correction = np.eye(affine_ndim + 1)
            for i, dim in enumerate(affine_sdims):
                scale = spacing_mem[dim] / spacing_zarr[dim]
                correction[i, i] = scale
                correction[i, affine_ndim] = (
                    origin_mem[dim] - origin_zarr[dim] * scale
                )
            affine_np = affine_np @ correction

            affine_ng = _affine_to_neuroglancer_source_transform(
                affine_np,
                sdims=sdims,
                output_spacing=spacing_isim,
            )
            affine_ndim = affine_ng.shape[-1] - 1
            full_affines[isim][-affine_ndim - 1 :, -affine_ndim - 1 :] = affine_ng
    else:
        full_affines = [None for _ in ome_zarr_urls]
        spacings_per_sim = [spacing] * len(ome_zarr_urls)

    if contrast_limits is not None:
        window = {
            "min": contrast_limits[0],
            "max": contrast_limits[1],
            "start": contrast_limits[0],
            "end": contrast_limits[1],
        }
        channel_index = 0
    # get contrast limits from first image
    elif "c" in dims:
        if channel_coord is None:
            channel_index = 0
        else:
            # this currently assumes that channel_coord
            # is present in all sims and at the same index
            channel_coord = str(channel_coord)
            channel_index = [str(c) for c in sim.coords["c"].values].index(
                channel_coord
            )
        if source_images is None:
            limits = np.array(
                [
                    sim_lims
                    for sim_lims in [
                        get_contrast_min_max_from_ome_zarr_omero_metadata(
                            path, channel_coord
                        )
                        for path in ome_zarr_paths
                    ]
                    if sim_lims is not None
                ]
            )
        else:
            limits = np.array([])
        if len(limits) == 0:
            window = None
        else:
            vmin, vmax = (float(v) for v in [np.min(limits), np.max(limits)])
            window = {
                "min": vmin,
                "max": vmax,
                "start": vmin,
                "end": vmax,
            }
    else:
        channel_index = 0
        window = None

    output_dimensions = {
        dim: [
            spacing[dim] if dim in sdims else 1,
            "um" if dim in sdims else "",
        ]
        for dim in dims
    }

    ng_config = {
        "dimensions": output_dimensions,
        "displayDimensions": sdims[::-1],
        "layerListPanel": {"visible": True},
        # 'position': [center[idim] for idim, dim in enumerate(sdims)],
        # "concurrentDownloads": 100, # leave at default
        "layout": "xy" if ndim == 2 else "4panel",
    }

    if not single_layer:
        ng_config["layers"] = [
            {
                # "type": "image",
                "source": {
                    "url": f"{url}",
                    "transform": {
                        # neuroglancer drops last row of homogeneous matrix
                        "matrix": [
                            [float(value) for value in row]
                            for row in full_affines[iview][:-1]
                        ],
                        "outputDimensions": {
                            (dim if dim != "c" else "c'"): [
                                spacings_per_sim[iview][dim]
                                if dim in sdims
                                else 1,
                                "um" if dim in sdims else "",
                            ]
                            for dim in dims
                        },
                    }
                    if full_affines[iview] is not None
                    else {},
                },
                "localDimensions": {"c'": [1, ""]} if "c" in dims else {},
                "localPosition": [channel_index] if "c" in dims else [],
                # 'localPosition': [0 for nsdim in nsdims] + [centers[iview][idim] for idim, dim in enumerate(sdims)],
                "tab": "rendering",
                "opacity": 0.6,
                # 'volumeRendering': 'on',
                "name": f"View {iview}",
            }
            | (
                {
                    "shaderControls": {
                        "normalized": {
                            "range": [window["min"], window["max"]],
                            "window": [window["start"], window["end"]],
                        },
                    },
                }
                if window is not None
                else {}
            )
            for iview, url in enumerate(ome_zarr_urls)
        ]

    else:
        ng_config["layers"] = [
            {
                # "type": "image",
                "source": [
                    {
                        "url": f"{url}",
                    }
                    | (
                        {
                            "transform": {
                                # neuroglancer drops last row of homogeneous matrix
                                "matrix": [
                                    [float(value) for value in row]
                                    for row in full_affines[iview][:-1]
                                ],
                                "outputDimensions": {
                                    (dim if dim != "c" else "c'"): [
                                        spacings_per_sim[iview][dim]
                                        if dim in sdims
                                        else 1,
                                        "um" if dim in sdims else "",
                                    ]
                                    for dim in dims
                                },
                            },
                        }
                        if full_affines[iview] is not None
                        else {}
                    )
                    for iview, url in enumerate(ome_zarr_urls)
                ],
                "localDimensions": {"c'": [1, ""]} if "c" in dims else {},
                "localPosition": [channel_index] if "c" in dims else [],
                "tab": "rendering",
                "opacity": 0.6,
                # 'volumeRendering': 'on',
                "name": "Tiles",
            }
            | (
                {
                    "shaderControls": {
                        "normalized": {
                            "range": [window["min"], window["max"]],
                            "window": [window["start"], window["end"]],
                        },
                    },
                }
                if window is not None
                else {}
            )
        ]

    # allow to overwrite / add settings for each layer
    if layer_dicts is not None:
        for il, layer_dict in enumerate(layer_dicts):
            ng_config["layers"][il] = {
                **ng_config["layers"][il],
                **layer_dict,
            }

    # allow to overwrite / add global settings
    if global_dict is not None:
        ng_config = {**ng_config, **global_dict}

    # import pprint
    # pprint.pprint(ng_config)
    return ng_config


def get_neuroglancer_url(ng_json):
    ng_url = "https://neuroglancer-demo.appspot.com/#!" + urllib.parse.quote(
        json.dumps(ng_json, separators=(",", ":"))
    )
    return ng_url


def view_neuroglancer(
    ome_zarr_paths=None,
    images=None,
    sims=None,
    transform_key=None,
    port=8000,
    channel_coord=None,
    single_layer=False,
    contrast_limits=None,
    layer_dicts: list[dict] = None,
    global_dict: dict = None,
):
    """
    Visualize a list of OME-Zarrs or in-memory images in Neuroglancer
    (browser-based, no installation required).

    If images (or sims) and transform_key are provided, the affine
    transformations attached to those images are used for visualization.
    Multi-scale images (msims / DataTree) are automatically reduced to their
    highest-resolution spatial image via ``msi_utils.get_sim_from_msim`` for
    transform lookup. If ``ome_zarr_paths`` is omitted, ``images`` are served
    directly as virtual read-only OME-Zarr 0.4 datasets.

    Confirmed to work with:
    - 2D and 3D
    - With affine transformations

    Parameters
    ----------
    ome_zarr_paths : list of str or Path, optional
        Paths to the OME-Zarr files to visualize. If omitted, ``images`` or
        ``sims`` are exposed through a virtual OME-Zarr server.
    images : list of spatial_image or DataTree, optional
        Spatial images (sims) or multi-scale images (msims) whose transform
        metadata is used for visualization.  msims are automatically converted
        to sims.  Takes precedence over ``sims`` when both are provided.
    sims : list of spatial_image, optional
        Kept for backward compatibility.  Use ``images`` for new code.
        Ignored when ``images`` is also provided.
    transform_key : str, optional
        Key of the affine transform to use for visualization.  Required when
        ``images`` or ``sims`` are provided.
    port : int, optional
        Port on which to serve local OME-Zarrs. By default 8000.
    channel_coord : str, optional
        Which channel to use for initializing contrast limits, by default None.
    single_layer : bool, optional
        Whether to show all images in a single layer (True) or in separate
        layers per image (False, default).
    contrast_limits : tuple, optional
        Contrast limits (min, max) to use for visualization. If None, limits
        are read from the OME-Zarr omero metadata if available, by default None.
    layer_dicts : list of dict, optional
        Per-layer neuroglancer config overrides.
    global_dict : dict, optional
        Global neuroglancer config overrides.
    """

    # Resolve the list of spatial images to use for transform lookup.
    # `images` takes precedence; fall back to the legacy `sims` parameter.
    if sims is not None and images is None:
        import warnings
        warnings.warn(
            "The 'sims' parameter is deprecated and will be removed in a future "
            "version. Use 'images' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    input_images = images if images is not None else sims
    if input_images is not None and not isinstance(input_images, (list, tuple)):
        input_images = [input_images]
    if input_images is not None:
        resolved_images = [
            msi_utils.get_sim_from_msim(img)
            if msi_utils.is_msim(img)
            else img
            for img in input_images
        ]
    else:
        resolved_images = None

    virtual_server = None
    source_images = None
    if ome_zarr_paths is None:
        if input_images is None:
            raise ValueError(
                "Either ome_zarr_paths or images must be provided."
            )

        virtual_msims = [
            img
            if msi_utils.is_msim(img)
            else msi_utils.get_msim_from_sim(img, scale_factors=[])
            for img in input_images
        ]
        virtual_server = ngff_utils.serve_virtual_ome_zarrs(
            virtual_msims,
            port=port,
        )
        ome_zarr_paths = virtual_server.urls
        ome_zarr_urls = [
            url.rstrip("/")
            for url in virtual_server.urls
        ]
        source_images = resolved_images
        dir_to_serve = None
    else:
        if isinstance(ome_zarr_paths, (str, os.PathLike)):
            ome_zarr_paths = [ome_zarr_paths]
        ome_zarr_paths = [str(p) for p in ome_zarr_paths]

        # determine a common root for all local paths so files in different
        # directories can all be served from a single HTTP server
        _MAX_SERVE_DEPTH = 3
        local_paths = [p for p in ome_zarr_paths if not p.startswith("http")]
        if local_paths:
            dir_to_serve = os.path.commonpath(
                [os.path.dirname(os.path.abspath(p)) for p in local_paths]
            )
            # safety check: refuse to serve overly broad directories
            for path in local_paths:
                rel = os.path.relpath(os.path.abspath(path), dir_to_serve)
                depth = len(rel.split(os.sep))
                if dir_to_serve == os.sep or depth > _MAX_SERVE_DEPTH:
                    import warnings

                    warnings.warn(
                        f"view_neuroglancer: the common ancestor directory "
                        f"'{dir_to_serve}' is too broad (depth {depth} > "
                        f"{_MAX_SERVE_DEPTH}) or is the filesystem root. "
                        f"Local files will not be served. "
                        f"Pass HTTP URLs instead.",
                        UserWarning,
                        stacklevel=2,
                    )
                    dir_to_serve = None
                    break
        else:
            dir_to_serve = None

        # generate urls for the ome zarr files
        # use forward slashes in URLs regardless of OS path separator
        ome_zarr_urls = [
            "http://localhost:{port}/{rel}".format(
                port=port,
                rel=os.path.relpath(
                    os.path.abspath(path), dir_to_serve
                ).replace(os.sep, "/"),
            )
            if (not path.startswith("http") and dir_to_serve is not None)
            else path
            for path in ome_zarr_paths
        ]

    ng_json = generate_neuroglancer_json(
        ome_zarr_paths=ome_zarr_paths,
        ome_zarr_urls=ome_zarr_urls,
        sims=resolved_images if transform_key is not None else None,
        source_images=source_images,
        transform_key=transform_key,
        channel_coord=channel_coord,
        single_layer=single_layer,
        contrast_limits=contrast_limits,
        layer_dicts=layer_dicts,
        global_dict=global_dict,
    )
    ng_url = get_neuroglancer_url(ng_json)

    print("Neuroglancer configuration JSON:")
    print(ng_json)

    print("Opening Neuroglancer in browser...")
    print("URL:", ng_url)
    print("Decoded URL:")
    print(urllib.parse.unquote(ng_url))
    print("Controls:")
    print("All panels")
    print("\t\tZoom: Ctrl + Mousewheel")
    print("Projection panels:")
    print("\t\tPan: Drag")
    print("\t\tRotate: Shift + Drag")
    print("3D view:")
    print("\t\tPan: Shift + Drag")
    print("\t\tRotate: Drag")

    webbrowser.open(ng_url)

    # serve the local or virtual OME-Zarr files
    if virtual_server is not None:
        virtual_server.serve_forever()
    elif dir_to_serve is not None:
        serve_dir(dir_to_serve, port=port)
