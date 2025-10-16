from multiview_stitcher import fusion, msi_utils

fused_sim = fusion.fuse(
    [msi_utils.get_sim_from_msim(msim) for msim in msims],
    transform_key="translation_registered",
)

# get fused array as a dask array
fused_sim.data

# get fused array as a numpy array
fused_sim.data.compute()