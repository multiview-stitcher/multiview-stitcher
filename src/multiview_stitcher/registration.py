import warnings

import dask.array as da
import networkx as nx
import numpy as np
import skimage.exposure
import skimage.registration
import skimage.registration._phase_cross_correlation  # skimage uses lazy importing
import xarray as xr
from dask import compute, delayed
from dask_image import ndfilters
from multiscale_spatial_image import MultiscaleSpatialImage
from scipy import ndimage
from tqdm import tqdm

from multiview_stitcher import (
    msi_utils,
    mv_graph,
    param_utils,
    spatial_image_utils,
    transformation,
)
from multiview_stitcher._skimage_monkey_patch import (
    _modified_disambiguate_shift,
)


class NotEnoughOverlapError(Exception):
    pass


def apply_recursive_dict(func, d):
    res = {}
    if isinstance(d, dict):
        for k, v in d.items():
            res[k] = apply_recursive_dict(func, v)
    else:
        return func(d)
    return res


def get_optimal_registration_binning(
    sim1,
    sim2,
    max_total_pixels_per_stack=(400) ** 3,
    use_only_overlap_region=False,
):
    """
    Heuristic to find good registration binning.

    - assume inputs contain only spatial dimensions.
    - assume inputs have same spacing
    - assume x and y have same spacing
    - so far, assume orthogonal axes

    Ideas:
      - sample physical space homogeneously
      - don't occupy too much memory

    """

    spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(sim1)
    ndim = len(spatial_dims)
    spacings = [
        spatial_image_utils.get_spacing_from_sim(sim, asarray=False)
        for sim in [sim1, sim2]
    ]

    registration_binning = {dim: 1 for dim in spatial_dims}

    if use_only_overlap_region:
        raise (NotImplementedError("use_only_overlap_region"))

    overlap = {
        dim: max(sim1.shape[idim], sim2.shape[idim])
        for idim, dim in enumerate(spatial_dims)
    }

    while (
        max(
            [
                np.product(
                    [
                        overlap[dim]
                        / spacings[isim][dim]
                        / registration_binning[dim]
                        for dim in spatial_dims
                    ]
                )
                for isim in range(2)
            ]
        )
        >= max_total_pixels_per_stack
    ):
        dim_to_bin = np.argmin(
            [
                min([spacings[isim][dim] for isim in range(2)])
                for dim in spatial_dims
            ]
        )

        if ndim == 3 and dim_to_bin == 0:
            registration_binning["z"] = registration_binning["z"] * 2
        else:
            for dim in ["x", "y"]:
                registration_binning[dim] = registration_binning[dim] * 2

        spacings = [
            {
                dim: spacings[isim][dim] * registration_binning[dim]
                for dim in spatial_dims
            }
            for isim in range(2)
        ]

    return registration_binning


def get_overlap_bboxes(
    sim1,
    sim2,
    input_transform_key=None,
    output_transform_key=None,
):
    """
    Get bounding box(es) of overlap between two spatial images
    in coord system given by input_transform_key, transformed
    into coord system given by output_transform_key (intrinsic
    coordinates if None).

    Return: lower and upper bounds of overlap for both input images
    """

    ndim = spatial_image_utils.get_ndim_from_sim(sim1)

    corners = [
        mv_graph.get_faces_from_sim(
            sim, transform_key=input_transform_key
        ).reshape(-1, ndim)
        for sim in [sim1, sim2]
    ]

    # project corners into intrinsic coordinate system
    corners_intrinsic = np.array(
        [
            [
                transformation.transform_pts(
                    corners[isim],
                    np.linalg.inv(
                        spatial_image_utils.get_affine_from_sim(
                            sim, transform_key=input_transform_key
                        ).data
                    ),
                )
                for isim in range(2)
            ]
            for sim in [sim1, sim2]
        ]
    )

    # project points into output_transform_key coordinate system if required
    if output_transform_key is not None:
        raise (NotImplementedError)
    else:
        corners_target_space = corners_intrinsic

    lowers = [
        np.max(np.min(corners_target_space[isim], axis=1), axis=0)
        for isim in range(2)
    ]
    uppers = [
        np.min(np.max(corners_target_space[isim], axis=1), axis=0)
        for isim in range(2)
    ]

    return lowers, uppers


def sims_to_intrinsic_coord_system(
    sim1,
    sim2,
    transform_key,
    overlap_bboxes,
):
    """
    Transform images into intrinsic coordinate system of fixed image.

    Return: transformed spatial images
    """

    reg_sims_b = [sim1, sim2]
    lowers, uppers = overlap_bboxes

    # get images into the same physical space (that of sim1)
    spatial_image_utils.get_ndim_from_sim(reg_sims_b[0])
    # spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(reg_sims_b[0])
    spacing = spatial_image_utils.get_spacing_from_sim(
        reg_sims_b[0], asarray=True
    )

    # transform images into intrinsic coordinate system of fixed image
    affines = [
        sim.attrs["transforms"][transform_key].squeeze().data
        for sim in reg_sims_b
    ]
    transf_affine = np.matmul(np.linalg.inv(affines[1]), affines[0])

    reg_sims_b_t = [
        transformation.transform_sim(
            sim,
            [None, transf_affine][isim],
            output_origin=lowers[0],
            output_spacing=spacing,
            output_shape=np.floor(
                np.array(uppers[0] - lowers[0]) / spacing + 1
            ).astype(np.uint16),
        )
        for isim, sim in enumerate(reg_sims_b)
    ]

    # attach transforms
    for _, sim in enumerate(reg_sims_b_t):
        spatial_image_utils.set_sim_affine(
            sim,
            spatial_image_utils.get_affine_from_sim(
                sim1, transform_key=transform_key
            ),
            transform_key=transform_key,
        )

    return reg_sims_b_t[0], reg_sims_b_t[1]


def phase_correlation_registration(
    sim1,
    sim2,
    transform_key,
    overlap_bboxes,
):
    """
    Perform phase correlation registration between two spatial images.
    """

    sims_intrinsic_cs = sims_to_intrinsic_coord_system(
        sim1,
        sim2,
        transform_key,
        overlap_bboxes,
    )

    ndim = spatial_image_utils.get_ndim_from_sim(sim1)
    spatial_image_utils.get_spacing_from_sim(sim1, asarray=True)

    def skimage_phase_corr_no_runtime_warning_and_monkey_patched(
        *args, **kwargs
    ):
        # monkey patching needs to be done here in order to work
        # with dask distributed
        skimage.registration._phase_cross_correlation._disambiguate_shift = (
            _modified_disambiguate_shift
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            result = skimage.registration.phase_cross_correlation(
                *args, **kwargs
            )
        return result

    shift = da.from_delayed(
        delayed(np.array)(
            # delayed(skimage.registration.phase_cross_correlation)(
            delayed(skimage_phase_corr_no_runtime_warning_and_monkey_patched)(
                delayed(lambda x: np.array(x))(sims_intrinsic_cs[0].data),
                delayed(lambda x: np.array(x))(sims_intrinsic_cs[1].data),
                upsample_factor=(10 if ndim == 2 else 2),
                disambiguate=True,
                normalization="phase",
            )[0]
        ),
        shape=(ndim,),
        dtype=float,
    )

    shift_matrix = param_utils.affine_from_translation(shift)

    # transform shift obtained in intrinsic coordinate system
    # into transformation between input images placed in
    # extrinsic coordinate system

    ext_param = get_affine_from_intrinsic_affine(
        data_affine=shift_matrix,
        sim_fixed=sims_intrinsic_cs[0],
        sim_moving=sims_intrinsic_cs[1],
        transform_key_fixed=transform_key,
        transform_key_moving=transform_key,
    )

    return param_utils.get_xparam_from_param(ext_param)


def get_affine_from_intrinsic_affine(
    data_affine,
    sim_fixed,
    sim_moving,
    transform_key_fixed=None,
    transform_key_moving=None,
):
    """
    Determine transform between extrinsic coordinate systems given
    a transform between intrinsic coordinate systems

    x_f_P = D_to_P_f * x_f_D

    x_f_D = M_D * x_c_D
    x_f_P = M_P * x_c_P
    x_f_W = M_W * x_c_W


    D_to_P_f * x_f_D = M_P * D_to_P_c * x_c_D
    x_f_D = inv(D_to_P_f) * M_P * D_to_P_c * x_c_D
    =>
    M_D = inv(D_to_P_f) * M_P * D_to_P_c
    =>
    D_to_P_f * M_D * inv(D_to_P_c) = M_P

    D_to_W_f * M_D * inv(D_to_W_c) = M_W

    x_f_P = D_to_P_f * x_f_D

    x_f_D = M_D * x_c_D
    x_f_P = M_P * x_c_P
    D_to_P_f * x_f_D = M_P * D_to_P_c * x_c_D
    x_f_D = inv(D_to_P_f) * M_P * D_to_P_c * x_c_D
    =>
    M_D = inv(D_to_P_f) * M_P * D_to_P_c
    =>
    D_to_P_f * M_D * inv(D_to_P_c) = M_P
    """

    if transform_key_fixed is None:
        phys2world_fixed = np.eye(data_affine.shape[0])
    else:
        phys2world_fixed = np.array(
            sim_fixed.attrs["transforms"][transform_key_moving]
        )

    if transform_key_moving is None:
        phys2world_moving = np.eye(data_affine.shape[0])
    else:
        phys2world_moving = np.array(
            sim_moving.attrs["transforms"][transform_key_moving]
        )

    D_to_P_f = np.matmul(
        param_utils.affine_from_translation(
            spatial_image_utils.get_origin_from_sim(sim_moving, asarray=True)
        ),
        np.diag(
            list(
                spatial_image_utils.get_spacing_from_sim(
                    sim_moving, asarray=True
                )
            )
            + [1]
        ),
    )
    P_to_W_f = phys2world_moving
    D_to_W_f = np.matmul(
        P_to_W_f,
        D_to_P_f,
    )

    D_to_P_c = np.matmul(
        param_utils.affine_from_translation(
            spatial_image_utils.get_origin_from_sim(sim_fixed, asarray=True)
        ),
        np.diag(
            list(
                spatial_image_utils.get_spacing_from_sim(
                    sim_fixed, asarray=True
                )
            )
            + [1]
        ),
    )
    P_to_W_c = phys2world_fixed
    D_to_W_c = np.matmul(
        P_to_W_c,
        D_to_P_c,
    )

    M_W = np.matmul(D_to_W_f, np.matmul(data_affine, np.linalg.inv(D_to_W_c)))

    return M_W


def register_pair_of_msims(
    msim1,
    msim2,
    transform_key=None,
    registration_binning=None,
    use_only_overlap_region=True,
    pairwise_reg_func=phase_correlation_registration,
    reg_func_kwargs=None,
):
    """
    Register the input images containing only spatial dimensions.

    Return: Transform in homogeneous coordinates.

    Parameters
    ----------
    msim1 : MultiscaleSpatialImage
        Fixed image.
    msim2 : MultiscaleSpatialImage
        Moving image.
    transform_key : str, optional
        Extrinsic coordinate system to consider as preregistration.
        This affects the calculation of the overlap region and is passed on to the
        registration func, by default None
    registration_binning : dict, optional
    use_only_overlap_region : bool, optional
        If True, only the precomputed overlap between images , by default True
    pairwise_reg_func : func, optional
        Function used for registration, which is passed as input two spatial images,
        a transform_key and precomputed bounding boxes. Returns a transform in
        homogeneous coordinates. By default phase_correlation_registration
    reg_func_kwargs : dict, optional
        Additional keyword arguments passed to the registration function

    Returns
    -------
    xarray.DataArray
        Transform in homogeneous coordinates mapping coordinates from the fixed
        to the moving image.
    """

    if reg_func_kwargs is None:
        reg_func_kwargs = {}
    spatial_dims = msi_utils.get_spatial_dims(msim1)
    # ndim = len(spatial_dims)

    sim1 = msi_utils.get_sim_from_msim(msim1)
    sim2 = msi_utils.get_sim_from_msim(msim2)

    lowers, uppers = get_overlap_bboxes(
        sim1,
        sim2,
        input_transform_key=transform_key,
        output_transform_key=None,
    )

    if use_only_overlap_region:
        reg_sims = [
            sim.sel(
                {
                    dim: slice(
                        lowers[isim][idim] - 0.001, uppers[isim][idim] + 0.001
                    )
                    for idim, dim in enumerate(spatial_dims)
                }
            )
            for isim, sim in enumerate([sim1, sim2])
        ]

    else:
        reg_sims = [sim1, sim2]

    if registration_binning is None:
        registration_binning = get_optimal_registration_binning(
            reg_sims[0], reg_sims[1]
        )

    if (
        registration_binning is not None
        and min(registration_binning.values()) > 1
    ):
        reg_sims_b = [
            sim.coarsen(registration_binning, boundary="trim")
            .mean()
            .astype(sim.dtype)
            for sim in reg_sims
        ]
    else:
        reg_sims_b = reg_sims

    # # Optionally perform CLAHE before registration
    # for i in range(2):
    #     # reg_sims_b[i].data = da.from_delayed(
    #     #     delayed(skimage.exposure.equalize_adapthist)(
    #     #         reg_sims_b[i].data, kernel_size=10, clip_limit=0.02, nbins=2 ** 13),
    #     #     shape=reg_sims_b[i].shape, dtype=float)

    #     reg_sims_b[i].data = da.map_overlap(
    #         skimage.exposure.equalize_adapthist,
    #         reg_sims_b[i].data,
    #         kernel_size=10,
    #         clip_limit=0.02,
    #         nbins=2 ** 13,
    #         depth={idim: 10 for idim, k in enumerate(spatial_dims)},
    #         dtype=float
    #         )

    return pairwise_reg_func(
        reg_sims_b[0],
        reg_sims_b[1],
        transform_key=transform_key,
        overlap_bboxes=(lowers, uppers),
        **reg_func_kwargs,
    )


def register_pair_of_msims_over_time(
    msim1,
    msim2,
    transform_key=None,
    registration_binning=None,
    use_only_overlap_region=True,
    pairwise_reg_func=phase_correlation_registration,
    reg_func_kwargs=None,
):
    """
    Apply register_pair_of_msims to each time point of the input images.
    """

    if reg_func_kwargs is None:
        reg_func_kwargs = {}
    msim1 = msi_utils.ensure_time_dim(msim1)
    msim2 = msi_utils.ensure_time_dim(msim2)

    sim1 = msi_utils.get_sim_from_msim(msim1)

    # suppress pandas future warning occuring within xarray.concat
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)

        xp = xr.concat(
            [
                register_pair_of_msims(
                    msi_utils.multiscale_sel_coords(msim1, {"t": t}),
                    msi_utils.multiscale_sel_coords(msim2, {"t": t}),
                    transform_key=transform_key,
                    registration_binning=registration_binning,
                    pairwise_reg_func=pairwise_reg_func,
                    reg_func_kwargs=reg_func_kwargs,
                    use_only_overlap_region=use_only_overlap_region,
                )
                for t in sim1.coords["t"].values
            ],
            dim="t",
        )

    xp = xp.assign_coords({"t": sim1.coords["t"].values})

    return xp


def get_registration_graph_from_overlap_graph(
    g,
    transform_key=None,
    registration_binning=None,
    use_only_overlap_region=True,
    pairwise_reg_func=phase_correlation_registration,
    reg_func_kwargs=None,
):
    if reg_func_kwargs is None:
        reg_func_kwargs = {}
    g_reg = g.to_directed()

    ref_node = mv_graph.get_node_with_masimal_overlap_from_graph(g)

    # invert overlap to use as weight in shortest path
    for e in g_reg.edges:
        g_reg.edges[e]["overlap_inv"] = 1 / (
            g_reg.edges[e]["overlap"] + 1
        )  # overlap can be zero

    # get shortest paths to ref_node
    paths = nx.shortest_path(g_reg, source=ref_node, weight="overlap_inv")

    # get all pairs of views that are connected by a shortest path
    for n, sp in paths.items():
        g_reg.nodes[n]["reg_path"] = sp

        if len(sp) < 2:
            continue

        # add registration edges
        for i in range(len(sp) - 1):
            pair = (sp[i], sp[i + 1])

            g_reg.edges[(pair[0], pair[1])]["transform"] = (
                register_pair_of_msims_over_time
            )(
                g.nodes[pair[0]]["msim"],
                g.nodes[pair[1]]["msim"],
                transform_key=transform_key,
                registration_binning=registration_binning,
                use_only_overlap_region=use_only_overlap_region,
                pairwise_reg_func=pairwise_reg_func,
                reg_func_kwargs=reg_func_kwargs,
            )

    g_reg.graph["pair_finding_method"] = "shortest_paths_considering_overlap"

    return g_reg


def get_node_params_from_reg_graph(g_reg):
    """
    Get final transform parameters by concatenating transforms
    along paths of pairwise affine transformations.
    """

    ndim = msi_utils.get_ndim(g_reg.nodes[list(g_reg.nodes)[0]]["msim"])

    for n in g_reg.nodes:
        reg_path = g_reg.nodes[n]["reg_path"]

        path_pairs = [
            [reg_path[i], reg_path[i + 1]] for i in range(len(reg_path) - 1)
        ]

        if "t" in g_reg.nodes[n]["msim"].dims:
            path_params = xr.DataArray(
                [np.eye(ndim + 1) for t in g_reg.nodes[n]["msim"].coords["t"]],
                dims=["t", "x_in", "x_out"],
                coords={
                    "t": msi_utils.get_sim_from_msim(
                        g_reg.nodes[n]["msim"]
                    ).coords["t"]
                },
            )
        else:
            path_params = param_utils.identity_transform(ndim)

        for pair in path_pairs:
            path_params = xr.apply_ufunc(
                np.matmul,
                g_reg.edges[(pair[0], pair[1])]["transform"],
                path_params,
                input_core_dims=[["x_in", "x_out"]] * 2,
                output_core_dims=[["x_in", "x_out"]],
                vectorize=True,
            )

        g_reg.nodes[n]["transforms"] = path_params

    return g_reg


def register(
    msims: [MultiscaleSpatialImage],
    reg_channel_index=None,
    transform_key=None,
    new_transform_key=None,
    registration_binning=None,
    use_only_overlap_region=True,
    pairwise_reg_func=phase_correlation_registration,
    reg_func_kwargs=None,
):
    """

    Register a list of views to a common extrinsic coordinate system.

    This function is the main entry point for registration.

    1) Build a graph of pairwise overlaps between views
    2) Determine registration pairs from this graph
    3) Register each pair of views.
       Need to add option to pass registration functions here.
    4) Determine the parameters mapping each view into the new extrinsic
       coordinate system.
       Currently done by determining a reference view and concatenating for reach
       view the pairwise transforms along the shortest paths towards the ref view.

    Parameters
    ----------
    msims : list of MultiscaleSpatialImage
        Input views
    reg_channel_index : int, optional
        Index of channel to be used for registration, by default None
    transform_key : str, optional
        Extrinsic coordinate system to use as a starting point
        for the registration, by default None
    new_transform_key : str, optional
        If set, the registration result will be registered as a new extrinsic
        coordinate system in the input views (with the given name), by default None
    registration_binning : dict, optional
        Binning applied to each dimensionn during registration, by default None
    pairwise_reg_func : func, optional
        Function used for registration.
    reg_func_kwargs : dict, optional
        Additional keyword arguments passed to the registration function

    Returns
    -------
    list of xr.DataArray
        Parameters mapping each view into a new extrinsic coordinate system
    """

    if reg_func_kwargs is None:
        reg_func_kwargs = {}
    sims = [msi_utils.get_sim_from_msim(msim) for msim in msims]

    if reg_channel_index is None:
        for msim in msims:
            if "c" in msi_utils.get_dims(msim):
                raise (Exception("Please choose a registration channel."))

    msims_reg = []
    for imsim, msim in enumerate(msims):
        if "c" in msi_utils.get_dims(msim):
            msim_reg = msi_utils.multiscale_sel_coords(
                msim, {"c": sims[imsim].coords["c"][reg_channel_index]}
            )
        else:
            msim_reg = msim
        msims_reg.append(msim_reg)

    g = mv_graph.build_view_adjacency_graph_from_msims(
        # [msi_utils.get_sim_from_msim(msim) for msim in msims_reg],
        msims_reg,
        transform_key=transform_key,
    )

    g_reg = get_registration_graph_from_overlap_graph(
        g,
        transform_key=transform_key,
        registration_binning=registration_binning,
        use_only_overlap_region=use_only_overlap_region,
        pairwise_reg_func=pairwise_reg_func,
        reg_func_kwargs=reg_func_kwargs,
    )

    if not len(g_reg.edges):
        raise (
            NotEnoughOverlapError(
                "Not enough overlap between views\
        for stitching. Consider stabilizing the tiles instead."
            )
        )

    g_reg_computed = mv_graph.compute_graph_edges(g_reg)

    g_reg_nodes = get_node_params_from_reg_graph(g_reg_computed)

    node_transforms = mv_graph.get_nodes_dataset_from_graph(
        g_reg_nodes, node_attribute="transforms"
    )
    params = [node_transforms[dv] for dv in node_transforms.data_vars]

    if new_transform_key is not None:
        for imsim, msim in enumerate(msims):
            msi_utils.set_affine_transform(
                msim,
                params[imsim],
                transform_key=new_transform_key,
                base_transform_key=transform_key,
            )

    return params


def stabilize(sims, reg_channel_index=0, sigma=2):
    sims = [sim.sel(c=sim.coords["c"][reg_channel_index]) for sim in sims]

    if len(sims[0].coords["t"]) < 8:
        raise (
            Exception("Need at least 8 time points to perform stabilization.")
        )

    params = [
        get_stabilization_parameters_from_sim(sim, sigma=sigma) for sim in sims
    ]

    params = compute(params)[0]

    return params


def correct_random_drift(
    ims,
    reg_ch=0,
    zoom_factor=10,
    particle_reinstantiation_stepsize=30,
    sigma_t=3,
):
    """
    ## Stage shift correction (currently 2d, but easily extendable)

    Goal: Correct for random stage shifts in timelapse movies in the absence of reference points that are fixed relative to the stage.

    Method: Assume that in the absence of random shifts particle trajectories are smooth in time. Find raw (uncorrected) trajectories, smooth them over time and determine the random stage shifts as the mean deviation of the actual trajectories from the smoothed ones.

    Steps:
    1) Calculate PIV fields (or optical flow) for the timelapse movie
    2) Track the initial coordinates as virtual particles throughout the timelapse
    3) Smooth trajectories and determine the mean deviations (N_t, d_x, d_y) from the actual trajectories
    4) Use the obtained deviations to correct the timelapse and verify result quality visually

    Comments:
    - assumes input data to be in the format (t, c, y, x)
    """

    import dask
    import dask.delayed as d
    from dask.diagnostics import ProgressBar

    # for some reason float images give a different output
    ims = ims.astype(np.uint16)

    of_channel = reg_ch
    regims = np.array(
        [
            ndimage.zoom(ims[it, of_channel], 1.0 / zoom_factor, order=1)
            for it in range(len(ims[:]))
        ]
    )

    fs = [
        d(skimage.registration.optical_flow_tvl1)(
            regims[t],
            regims[t + 1],
            attachment=10000,
        )
        for t in tqdm(range(len(ims) - 1)[:])
    ]

    print("Computing optical flow...")
    with ProgressBar():
        fs = np.array(dask.compute(fs)[0])  # *zoom_factor

    print("Tracking virtual particles...")

    # coordinates to be tracked
    x, y = np.mgrid[0 : regims[0].shape[0], 0 : regims[0].shape[1]]

    # define starting point(s) for coordinates to be tracked
    # for long movies, it's good to have several starting tps
    # as the coordinates can move out of the FOV

    starting_tps = range(0, len(ims), particle_reinstantiation_stepsize)

    posss = []
    for starting_tp in starting_tps:
        poss = [np.array([x, y])]
        for t in tqdm(range(starting_tp, len(ims) - 1)):
            displacement = np.array(
                [
                    ndimage.map_coordinates(
                        fs[t][dim],
                        np.array(
                            [poss[-1][0].flatten(), poss[-1][1].flatten()]
                        ),
                        order=1,
                        mode="constant",
                        cval=np.nan,
                    ).reshape(x.shape)
                    for dim in range(2)
                ]
            )
            poss.append(displacement + poss[-1])
        posss.append(np.array(poss))

    print("Smoothing trajectories...")
    posss_smooth = [
        ndimage.gaussian_filter(
            posss[istp], [sigma_t, 0, 0, 0], mode="nearest"
        )
        for istp, starting_tp in enumerate(starting_tps)
    ]

    devs = np.array(
        [
            np.nanmean(
                [
                    posss_smooth[istp][t - starting_tp]
                    - posss[istp][t - starting_tp]
                    for istp, starting_tp in enumerate(starting_tps)
                    if starting_tp <= t
                ],
                axis=(0, -1, -2),
            )
            for t in range(len(ims))
        ]
    )

    devs = devs * zoom_factor

    print("Correct drifts...")
    imst = np.array(
        [
            [
                ndimage.affine_transform(
                    ims[t, ch],
                    matrix=[[1, 0], [0, 1]],
                    offset=-devs[t],
                    order=1,
                )
                for ch in range(ims.shape[1])
            ]
            for t in tqdm(range(len(ims)))
        ]
    )

    return imst, devs


def get_stabilization_parameters(tl, sigma=2):
    """

    Correct for random stage shifts in timelapse movies in the absence of reference points that are fixed relative to the stage.
    - obtain shifts between consecutive frames
    - get cumulative sum of shifts
    - smooth
    - consider difference between smoothed and unsmoothed shifts as the random stage shifts

    Assume first dimension is time.

    tl: dask array of shape (N_T, ...)
    """

    ndim = tl[0].ndim

    ps = da.stack(
        [
            da.from_delayed(
                delayed(skimage.registration.phase_cross_correlation)(
                    tl[t - 1], tl[t], upsample_factor=10, normalization=None
                )[0],
                shape=(ndim,),
                dtype=float,
            )
            for t in range(1, tl.shape[0])
        ]
    )

    ps = da.concatenate([da.zeros((1, ndim)), ps], axis=0)

    ps_cum = da.cumsum(ps, axis=0)
    ps_cum_filtered = ndfilters.gaussian_filter(
        ps_cum, [sigma, 0], mode="nearest"
    )
    deltas = ps_cum - ps_cum_filtered
    deltas = -deltas

    # tl_t = da.stack([da.from_delayed(delayed(ndimage.affine_transform)(tl[t],
    #                                         matrix=np.eye(2),
    #                                         offset=-params[t], order=1), shape=tl[0].shape, dtype=tl[0].dtype)
    #             for t in range(N_t)]).compute()

    return deltas


def get_stabilization_parameters_from_sim(sim, sigma=2):
    spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(sim)

    params = get_stabilization_parameters(
        sim.transpose(*tuple(["t"] + spatial_dims)), sigma=sigma
    )

    params = [
        param_utils.affine_from_translation(
            params[it]
            * spatial_image_utils.get_spacing_from_sim(sim, asarray=True)
        )
        for it, t in enumerate(sim.coords["t"])
    ]

    # suppress pandas future warning occuring within xarray.concat
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)

        params = xr.concat(
            [param_utils.get_xparam_from_param(p) for p in params], dim="t"
        )

    params = params.assign_coords({"t": sim.coords["t"]})

    return params


def get_drift_correction_parameters(tl, sigma=2):
    """
    Assume first dimension is time

    tl: dask array of shape (N_T, ...)
    """

    ndim = tl[0].ndim

    ps = da.stack(
        [
            da.from_delayed(
                delayed(skimage.registration.phase_cross_correlation)(
                    tl[t - 1], tl[t], upsample_factor=2, normalization=None
                )[0],
                shape=(ndim,),
                dtype=float,
            )
            for t in range(1, tl.shape[0])
        ]
    )

    ps = da.concatenate([da.zeros((1, ndim)), ps], axis=0)

    ps_cum = -da.cumsum(ps, axis=0)

    # ps_cum_filtered = ndfilters.gaussian_filter(ps_cum, [sigma, 0], mode='nearest')
    # deltas = ps_cum_filtered - ps_cum

    # tl_t = da.stack([da.from_delayed(delayed(ndimage.affine_transform)(tl[t],
    #                                         matrix=np.eye(2),
    #                                         offset=-params[t], order=1), shape=tl[0].shape, dtype=tl[0].dtype)
    #             for t in range(N_t)]).compute()

    return ps_cum


def crop_sim_to_references(
    sim_input_to_crop,
    reference_sims,
    transform_key_input,
    transform_keys_reference,
    input_time_index=0,
):
    """
    Crop input image to the minimal region fully covering the reference sim(s).
    """

    ref_corners_world = []
    for irefsim, reference_sim in enumerate(reference_sims):
        ref_corners_world += list(
            np.unique(
                mv_graph.get_faces_from_sim(
                    reference_sim,
                    transform_key=transform_keys_reference[irefsim],
                ).reshape((-1, 2)),
                axis=0,
            )
        )

    input_affine = spatial_image_utils.get_affine_from_sim(
        sim_input_to_crop, transform_key=transform_key_input
    )

    if "t" in input_affine.dims:
        input_affine = input_affine.sel(
            {"t": input_affine.coords["t"][input_time_index]}
        )

    input_affine_inv = np.linalg.inv(np.array(input_affine))
    ref_corners_input_dataspace = transformation.transform_pts(
        ref_corners_world, np.array(input_affine_inv)
    )

    lower, upper = (
        func(ref_corners_input_dataspace, axis=0) for func in [np.min, np.max]
    )

    sdims = spatial_image_utils.get_spatial_dims_from_sim(sim_input_to_crop)

    sim_cropped = spatial_image_utils.sim_sel_coords(
        sim_input_to_crop,
        {
            dim: (sim_input_to_crop.coords[dim] > lower[idim])
            * (sim_input_to_crop.coords[dim] < upper[idim])
            for idim, dim in enumerate(sdims)
        },
    )

    return sim_cropped
