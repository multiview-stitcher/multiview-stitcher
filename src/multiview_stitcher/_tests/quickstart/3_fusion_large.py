from multiview_stitcher import fusion, misc_utils

output_zarr_url = "fused_output.ome.zarr"

fused = fusion.fuse_to_multiscale_ome_zarr(
    fuse_kwargs={
        "sims": [msi_utils.get_sim_from_msim(msim) for msim in msims],
        "transform_key": "translation_registered",
        # ... further optional args for fusion.fuse
    },
    output_zarr_url=output_zarr_url,
    # optionally, we can use ray for parallelization (`pip install "ray[default]"`)
    # batch_func=misc_utils.process_batch_using_ray,
    # n_batch=4, # number of chunk fusions to schedule / submit at a time
    # batch_func_kwargs={
    #     'num_cpus': 4 # number of processes for parallel processing to use with ray
    #     },
)