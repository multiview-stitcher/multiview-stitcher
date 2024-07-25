from dask.diagnostics import ProgressBar
from multiview_stitcher import registration

with ProgressBar():
    params = registration.register(
        msims,
        reg_channel="DAPI",  # channel to use for registration
        transform_key="stage_metadata",
        new_transform_key="translation_registered",
    )

# plot the tile configuration after registration
# vis_utils.plot_positions(msims, transform_key='translation_registered', use_positional_colors=False)
