from multiview_stitcher import fusion

fused_msim = fusion.fuse(
    images=msims,
    transform_key="translation_registered",
)

# get fused array at the highest output resolution as a dask array
fused_msim["scale0/image"].data

# get fused array at the highest output resolution as a numpy array
fused_msim["scale0/image"].data.compute()