import json
import os
import urllib
import webbrowser
from http.server import (
    HTTPServer,
    SimpleHTTPRequestHandler,
    test,
)

import numpy as np
import zarr
from matplotlib import colormaps, colors
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from ome_zarr.io import parse_url
from xarray import DataTree

from multiview_stitcher import (
    msi_utils,
    mv_graph,
    ngff_utils,
    spatial_image_utils,
)


def plot_positions(
    sims,
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
    plot_title=None,
    spacing=None,
    output_filename=None,
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
    spacing : dict, optional
        Overwrite the sims' spacing for plotting. Useful in the case of images with single
        coordinates for which the spacing is not defined in the metadata.
        By default None
    plot_title : str, optional
        Title of the plot, by default no title
    output_filename : str, optional
        Filename where to save the plot if not None, by default None

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


def plot_stack_props(stack_props, ax, color="black", size=10, linewidth=1):
    ndim = mv_graph.get_ndim_from_stack_props(stack_props)
    faces = mv_graph.get_faces_from_stack_props(stack_props)

    # get line segments
    line_segments = []

    if ndim == 3:
        for face in faces:
            inds = np.argsort(np.linalg.norm(face[1:] - face[0], axis=-1))
            line_segments.append([face[0], face[inds[0] + 1]])
            line_segments.append([face[0], face[inds[1] + 1]])
            line_segments.append(
                [
                    face[inds[0] + 1],
                    face[inds[0] + 1] + face[inds[1] + 1] - face[0],
                ]
            )
            line_segments.append(
                [
                    face[inds[1] + 1],
                    face[inds[1] + 1] + face[inds[0] + 1] - face[0],
                ]
            )

    elif ndim == 2:
        for face in faces:
            line_segments.append([face[0], face[1]])

    line_segments = np.array(line_segments)

    if ndim == 2:
        line_segments = np.concatenate(
            [np.zeros((line_segments.shape[0], 2, 1)), line_segments], axis=-1
        )

    line_collection = Line3DCollection(
        line_segments[:, :, [0, 2, 1]], colors=color, linewidths=linewidth
    )

    ax.add_collection3d(line_collection)


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
    ome_zarr_path, channel_label=None
):
    """
    Get contrast limits from the OME-Zarr omero metadata key
    for a specific channel. If channel_label is None, the
    first channel is used.
    """
    store = parse_url(ome_zarr_path, mode="r").store
    root = zarr.group(store=store)

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


def generate_neuroglancer_json(
    ome_zarr_paths,
    ome_zarr_urls,
    sims=None,
    transform_key=None,
    channel_coord=None,
    single_layer=False,
):
    # read the first multiscales
    sim = ngff_utils.read_sim_from_ome_zarr(ome_zarr_paths[0])
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
        for isim, sim in enumerate(sims):
            sim_spacing = spatial_image_utils.get_spacing_from_sim(sim)
            affine = spatial_image_utils.get_affine_from_sim(
                sim, transform_key=transform_key
            )
            if "t" in affine.dims:
                affine = affine.sel(t=0)
            affine = np.array(affine)
            affine_ndim = affine.shape[-1] - 1
            # https://github.com/google/neuroglancer/issues/538
            affine[:-1, -1] = affine[:-1, -1] / np.array(
                [sim_spacing[dim] for dim in sdims]
            )
            full_affines[isim][-affine_ndim - 1 :, -affine_ndim - 1 :] = affine
    else:
        full_affines = [None for _ in ome_zarr_paths]

    # get contrast limits from first image
    if "c" in dims:
        if channel_coord is None:
            channel_index = 0
        else:
            # this currently assumes that channel_coord
            # is present in all sims and at the same index
            channel_coord = str(channel_coord)
            channel_index = [str(c) for c in sim.coords["c"].values].index(
                channel_coord
            )
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

    # import pprint
    # pprint.pprint(ng_config)
    return ng_config


def get_neuroglancer_url(ng_json):
    ng_url = "https://neuroglancer-demo.appspot.com/#!" + urllib.parse.quote(
        json.dumps(ng_json, separators=(",", ":"))
    )
    return ng_url


def view_neuroglancer(
    ome_zarr_paths,
    sims=None,
    transform_key=None,
    port=8000,
    channel_coord=None,
    single_layer=False,
):
    """
    Visualize a list of OME-zarrs in Neuroglancer
    (browser-based, no installation required).

    If sims and transform_key are provided, the affine transformations saved
    in the sims are used for visualization.

    Confirmed to work with:
    - 2D and 3D
    - With affine transformations

    Parameters
    ----------
    ome_zarr_paths : list of str
        path to OME-Zarrs
    sims : list of spatial_images, optional
    transform_key : str, optional
        transform_key to use for visualization
    port : int, optional
        Port to serve OME-Zarrs. By default 8000
    channel_coord : str, optional
        Which channel to use for initializing contrast limits, by default None
    """

    # generate urls for the ome zarr files
    ome_zarr_urls = [
        f"http://localhost:{port}/{os.path.basename(path)}"
        if not path.startswith("http")
        else path
        for path in ome_zarr_paths
    ]

    ng_json = generate_neuroglancer_json(
        ome_zarr_paths=ome_zarr_paths,
        ome_zarr_urls=ome_zarr_urls,
        sims=sims,
        transform_key=transform_key,
        channel_coord=channel_coord,
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

    # serve the local ome-zarr files
    local = False
    for path in ome_zarr_paths:
        if not path.startswith("http"):
            dir_to_serve = os.path.dirname(path)
            local = True
            break

    if local:
        serve_dir(dir_to_serve, port=port)
