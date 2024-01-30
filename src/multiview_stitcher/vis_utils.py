from Geometry3D import Visualizer
from matplotlib import pyplot as plt

from multiview_stitcher import msi_utils, mv_graph, spatial_image_utils


def plot_positions(
    msims, transform_key, use_positional_colors=True, n_colors=2, t=None
):
    ndim = msi_utils.get_ndim(msims[0])

    sims = [msi_utils.get_sim_from_msim(msim) for msim in msims]

    if "t" in sims[0].dims and len(sims[0].coords["t"]) > 1:
        if t is None:
            t = sims[0].coords["t"][0]
        sims = [
            spatial_image_utils.sim_sel_coords(sim, {"t": t}) for sim in sims
        ]

    if use_positional_colors:
        colors = ["red", "green", "blue", "yellow"]
        greedy_colors = mv_graph.get_greedy_colors(
            # [spatial_image_utils.sim_sel_coords(
            #     sim, {'t':sim.coords['t'][0]})
            #     for sim in sims],
            sims,
            n_colors=n_colors,
            transform_key=transform_key,
        )
        colors = [
            colors[greedy_colors[iview] % len(colors)]
            for iview in range(len(msims))
        ]

    else:
        colors = ["red"] * len(msims)

    v = Visualizer(backend="matplotlib")

    for iview, sim in enumerate(sims):
        view_domain = mv_graph.get_poly_from_sim(
            sim, transform_key=transform_key
        )

        v.add((view_domain, colors[iview], 1))

    fig, ax = show_geometry3d_visualizer(v)

    ax.set_xlabel("z [μm]")
    ax.set_ylabel("y [μm]")
    ax.set_zlabel("x [μm]")

    ax.set_aspect("equal", adjustable="box")

    if ndim == 3:
        ax.view_init(elev=15, azim=-15, roll=0)
    elif ndim == 2:
        ax.view_init(elev=0, azim=0, roll=0)

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
        ax.scatter(point.x, point.y, point.z, c=color, s=size)

    for segment_tuple in self.segment_set:
        segment = segment_tuple[0]
        color = segment_tuple[1]
        size = segment_tuple[2]
        x = [segment.start_point.x, segment.end_point.x]
        y = [segment.start_point.y, segment.end_point.y]
        z = [segment.start_point.z, segment.end_point.z]
        ax.plot(x, y, z, color=color, linewidth=size)

    for arrow_tuple in self.arrow_set:
        x, y, z, u, v, w, length = arrow_tuple[0].get_tuple()
        color = arrow_tuple[1]
        size = arrow_tuple[1]
        ax.quiver(x, y, z, u, v, w, color=color, length=length)

    return fig, ax
