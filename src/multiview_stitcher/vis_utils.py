import numpy as np
from matplotlib import colormaps, colors
from matplotlib import pyplot as plt

from multiview_stitcher import msi_utils, mv_graph, spatial_image_utils
from multiview_stitcher.misc_utils import DisableLogger

with DisableLogger():
    from Geometry3D import Visualizer

import json
import os
import urllib
import webbrowser
from http.server import (
    HTTPServer,
    SimpleHTTPRequestHandler,
    test,
)


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
    view_labels=None,
    view_labels_size=10,
    show_plot=True,
    output_filename=None,
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
    view_labels : list of str, optional
        Custom labels to use for the views, by default None
    view_labels_size : int, optional
        Size of the view labels, by default 10
    show_plot : bool, optional
        Whether to show the plot, by default True
    output_filename : str, optional
        Filename where to save the plot if not None, by default None

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

            edge_cmap.set_bad(color="gray")

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
    if output_filename is not None:
        plt.savefig(output_filename)
    if show_plot:
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


def get_contrast_min_max_from_ome_zarr_omero_metadata(
    zarr_path, channel_label
):
    with open(os.path.join(zarr_path, ".zattrs")) as f:
        metadata = json.load(f)

    channel_matches = [
        ic
        for ic, c in enumerate(metadata["omero"]["channels"])
        if str(c["label"]) == str(channel_label)
    ]

    if not len(channel_matches) == 1:
        raise ValueError(
            f"Channel {channel_label} not found in metadata in {zarr_path}"
        )
    else:
        channel_index = channel_matches[0]
    window = metadata["omero"]["channels"][channel_index]["window"]

    return np.array([window["start"], window["end"]])


def generate_neuroglancer_json(
    sims,
    zarr_paths,
    zarr_urls,
    transform_key,
    channel_coord=None,
    single_layer=False,
):
    dims = sims[0].dims
    sdims = spatial_image_utils.get_spatial_dims_from_sim(sims[0])
    ndim = len(sdims)
    spacing = spatial_image_utils.get_spacing_from_sim(sims[0])

    full_affines = [np.eye(len(dims) + 1) for _ in sims]
    for isim, sim in enumerate(sims):
        affine = spatial_image_utils.get_affine_from_sim(
            sim, transform_key=transform_key
        )
        if "t" in affine.dims:
            affine = affine.sel(t=0)
        affine = np.array(affine)
        affine_ndim = affine.shape[-1] - 1
        # https://github.com/google/neuroglancer/issues/538
        affine[:-1, -1] = affine[:-1, -1] / np.array(
            [spacing[dim] for dim in sdims]
        )
        full_affines[isim][-affine_ndim - 1 :, -affine_ndim - 1 :] = affine

    # get contrast limits from first image
    if "c" in dims:
        if channel_coord is None:
            channel_coord = str(sims[0].coords["c"].values[0])
        else:
            channel_coord = str(channel_coord)
        channel_index = [str(c) for c in sims[0].coords["c"].values].index(
            channel_coord
        )
        limits = np.array(
            [
                get_contrast_min_max_from_ome_zarr_omero_metadata(
                    zarr_path, str(channel_coord)
                )
                for zarr_path in zarr_paths
            ]
        )
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
        dim: [spacing[dim] * 1e-6 if dim in sdims else 1.0, ""] for dim in dims
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
                "type": "image",
                "source": {
                    "url": f"zarr://{url}",
                    "transform": {
                        # neuroglancer drops last row of homogeneous matrix
                        "matrix": [
                            list(row) for row in full_affines[iview][:-1]
                        ],
                        "outputDimensions": {
                            (dim if dim != "c" else "c'"): d
                            for dim, d in output_dimensions.items()
                        },
                    },
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
            for iview, url in enumerate(zarr_urls)
        ]

    else:
        ng_config["layers"] = [
            {
                "type": "image",
                "source": [
                    {
                        "url": f"zarr://{url}",
                        "transform": {
                            # neuroglancer drops last row of homogeneous matrix
                            "matrix": [
                                list(row) for row in full_affines[iview][:-1]
                            ],
                            "outputDimensions": {
                                (dim if dim != "c" else "c'"): d
                                for dim, d in output_dimensions.items()
                            },
                        },
                    }
                    for iview, url in enumerate(zarr_urls)
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

    # import pprint
    # pprint.pprint(ng_config)
    return ng_config


def get_neuroglancer_url(ng_json):
    ng_url = "https://neuroglancer-demo.appspot.com/#!" + urllib.parse.quote(
        json.dumps(ng_json, separators=(",", ":"))
    )
    return ng_url


def view_neuroglancer(
    sims,
    zarr_paths,
    transform_key,
    port=8000,
    channel_coord=None,
    single_layer=False,
):
    """
    Visualize a list of spatial_images together with their OME-Zarrs on disk
    in Neuroglancer. No installation of Neuroglancer is required.

    TODO:
    - check 2D
    - check transforms for full affine

    Parameters
    ----------
    sims : list of spatial_images
    zarr_paths : list of str
        path to OME-Zarrs that have previously been saved from the sims
    transform_key : str
        transform_key to use for visualization
    port : int, optional
        Port to serve OME-Zarrs. By default 8000
    channel_coord : str, optional
        Which channel to use for initializing contrast limits, by default None
    """

    zarr_urls = [
        f"http://localhost:{port}/{os.path.basename(zarr_path)}"
        for zarr_path in zarr_paths
    ]

    ng_json = generate_neuroglancer_json(
        sims,
        zarr_paths,
        zarr_urls,
        channel_coord=channel_coord,
        transform_key=transform_key,
        single_layer=single_layer,
    )
    ng_url = get_neuroglancer_url(ng_json)

    print("Opening Neuroglancer in browser...")
    print("URL:", ng_url)
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

    parent_dir, image_name = os.path.split(zarr_paths[0])
    parent_dir = str(parent_dir)

    # serve the zarr files
    serve_dir(parent_dir, port=port)
