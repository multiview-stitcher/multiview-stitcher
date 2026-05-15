from multiview_stitcher import fusion

fused = fusion.fuse(
    sims=[msi_utils.get_sim_from_msim(msim) for msim in msims],
    transform_key="translation_registered",
    # ... further optional args for fusion.fuse
    output_zarr_url="fused_output.ome.zarr",
    zarr_options={
        "ome_zarr": True,
        # "ngff_version": "0.4",  # optional
    },
    # optionally, we can use joblib for parallelization (`pip install joblib` and `from multiview_stitcher import misc_utils`):
    # batch_options={
    #     "batch_func": misc_utils.process_batch_using_joblib,
    #     "n_batch": 20,  # number of chunk fusions to schedule / submit at a time
    #     "batch_func_kwargs": {
    #         'n_jobs': 4  # number of parallel jobs for joblib
    #     },
    # },
)