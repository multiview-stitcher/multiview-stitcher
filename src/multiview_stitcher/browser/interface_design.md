# Web app interface design

The interface should
- be simple and intuitive
- be responsive (should work on different screen sizes)

## Upper panel

- left: multiview stitcher logo / app name
- center: progress bar
- right:
  - "Log" button to show log
  - spin box for number of workers, by default 3
  - Viewer controls help button
    - zoom in/out
    - pan
  - github link to multiview-stitcher repo
  - "About" button
    - neuroglancer, ome-zarr, pyodide

## Center panel

- neuroglancer viewer
- for yx data, the layout should be 'xy'
- for zyx data, the layout should be 4 panels
- don't show layer or shader control panels

## Left panel

Data visualization and control panel

- Data drop zone
  - drag and drop OME-Zarrs (top level ome-zarr or folder containing multiple ome-zarrs)
  - click to open file dialog
- List of loaded msims
  - elements on each msim:
    - Remove
    - Short info (shape per dim, number of res levels)
    - Visibility toggle
- Coordinate system selection
  - dropdown menu to select transform_key to show in neuroglancer viewer
- Display options
  - drop down for which channel(s) to show in neuroglancer viewer: 
    - "Show all channels"
    - <name of channel 1>
    - <name of channel 2>
    ...
  - contrast limits (min, max), set for all channels together

## Right panel

Data manipulation and computation panel

Different tabs for different types of operations:

- Interactive tile placement
  - "New transform_key" button to create a new transform_key, with text for user to enter name of new transform_key (by default, the new transform_key is created as a copy of the currently selected transform_key)

- Registration
  - sub tabs:
    - Common options
      - drop down: Registration channel
      - Text field: New transform_key name, default to "registered"
    - Advanced options:
      - registration binning
  - "Register" button to run registration

- Fusion
  - sub tabs:
    - Common options
      - Fusion method
    - Advanced options:
      - blending widths
      - output spacing
  - "Fuse (preview)"
  - "Fuse to OME-Zarr"

  ## Notes

  - The basis for data manipulation is the currently selected transform_key