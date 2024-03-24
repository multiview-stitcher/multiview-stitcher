import numpy as np
from matplotlib import cm, colors
from matplotlib import pyplot as plt

from multiview_stitcher import msi_utils, mv_graph, spatial_image_utils
from multiview_stitcher.misc_utils import DisableLogger

with DisableLogger():
    from Geometry3D import Visualizer


def plot_positions(
    msims,
    transform_key,
    edges=None,
    edge_color_vals=None,
    edge_cmap=None,
    edge_clims=None,
    edge_label="edge weight",
    use_positional_colors=True,
    n_colors=2,
    nscoord=None,
    display_view_indices=True,
):
    """
    Plot tile / view positions in both 2D or 3D.

    Parameters
    ----------
    msims : list of multiscale_spatial_image (multiview-stitcher flavor)
        _description_
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

    Returns
    -------
    fig, ax : matplotlib figure and axis
    """

    if nscoord is None:
        nscoord = {}

    ndim = msi_utils.get_ndim(msims[0])

    sims = [msi_utils.get_sim_from_msim(msim) for msim in msims]

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
            for iview in range(len(msims))
        ]

    else:
        pos_colors = ["black"] * len(msims)

    v = Visualizer(backend="matplotlib")

    for iview, sim in enumerate(sims):
        view_domain = mv_graph.get_poly_from_stack_props(
            spatial_image_utils.get_stack_properties_from_sim(
                sim, transform_key=transform_key
            )
        )

        v.add((view_domain, pos_colors[iview], 1))

    fig, ax = show_geometry3d_visualizer(v)

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
            ax.text(z, x, y, str(iview), size=10, zorder=1, color="k")

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
                edge_cmap = cm.get_cmap(
                    "Spectral",
                )

            edge_colors = [edge_cmap(norm(val)) for val in edge_color_vals]
        else:
            edge_colors = ["k" for _ in edges]

        # Plot the edges
        for e, color in zip(edges, edge_colors):
            ax.plot(
                *np.array([node_poss_mpl[e[0]], node_poss_mpl[e[1]]]).T,
                linestyle="--",
                color=color,
            )

        if edge_color_vals is not None:
            sm = plt.cm.ScalarMappable(cmap=edge_cmap)
            sm.set_array(
                list(edge_color_vals) + [edge_clims[0], edge_clims[1]]
            )
            plt.colorbar(sm, label=edge_label, ax=ax)

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

    plt.tight_layout()
    plt.show()

    return fig, ax


def show_geometry3d_visualizer(self):
    """
    This method replaces the show method of the Geometry3D Visualizer class
    and
    - uses `ax = Axes3D(fig)` instead of `ax = fig.add_subplot(projection='3d')`
    - sets the aspect ratio to 'equal'
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_aspect("equal")
    for point_tuple in self.point_set:
        point = point_tuple[0]
        color = point_tuple[1]
        size = point_tuple[2]
        ax.scatter(point.x, point.z, point.y, c=color, s=size)

    for segment_tuple in self.segment_set:
        segment = segment_tuple[0]
        color = segment_tuple[1]
        size = segment_tuple[2]
        x = [segment.start_point.x, segment.end_point.x]
        y = [segment.start_point.y, segment.end_point.y]
        z = [segment.start_point.z, segment.end_point.z]
        ax.plot(x, z, y, color=color, linewidth=size)

    return fig, ax
