from multiview_stitcher import fusion, misc_utils

fused = fusion.fuse(
    sims=[msi_utils.get_sim_from_msim(msim) for msim in msims],
    transform_key="translation_registered",
    # ... further optional args for fusion.fuse
    output_zarr_url="fused_output.ome.zarr",
    zarr_options={
        "ome_zarr": True,
        # "ngff_version": "0.4",  # optional
    },
    # optionally, we can use ray for parallelization (`pip install "ray[default]"`)
    # batch_options={
    #     "batch_func": misc_utils.process_batch_using_ray,
    #     "n_batch": 4,  # number of chunk fusions to schedule / submit at a time
    #     "batch_func_kwargs": {
    #         'num_cpus': 4  # number of processes for parallel processing to use with ray
    #     },
    # },
)