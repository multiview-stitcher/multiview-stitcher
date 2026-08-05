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
    - zoom in/out (requires ctrl)
    - pan
  - github link to multiview-stitcher repo
  - "About" button
    - neuroglancer, ome-zarr, pyodide

## Center panel

- neuroglancer viewer
- for yx data, the layout should be 'xy'
- for zyx data, the layout should be 4 panels
- don't show an open layer panel, but don't make it inaccessible
- don't show the tool palette

## Left panel

Data visualization and control panel

- Data drop zone
  - drag and drop OME-Zarrs (top level ome-zarr or folder containing multiple ome-zarrs)
  - click to open file dialog
- List of loaded msims + fused preview (if available)
  - elements on each msim:
    - Remove
    - Short info (shape per dim, number of res levels)
    - Visibility toggle
- Coordinate system selection
  - dropdown menu to select transform_key to show in neuroglancer viewer
- Display options
  - list of channels
    - "Positional colors": checkbox to color channels based on their position (see vis_utils.py for details), fused image is unaffected by this option
    - for each channel:
      - visibility toggle
      - contrast limits (min, max)
        - double range slider
        - in the same line / height: min / max text fields

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
      - output chunksizes
  - "Fuse (preview)"
  - "Fuse to OME-Zarr"

## Notes

- The basis for data manipulation is the currently selected transform_key

# Testing

A short retrospective on what actually earned its keep.

## The harness that found most bugs

A headless Chromium page loading the **real `viewer.js`** against **real Python-generated states**, with fixtures dumped byte-for-byte from `session.serve()` so a static file server was indistinguishable from the running app. Cheap to build, and it reproduced every viewer bug on the first try.

**Probe internals, not pixels.** Software GL here can't give Neuroglancer its framebuffer, so screenshots were useless. Reading `loadState`, coordinate-space names/bounds, and `navigationState.position` was strictly better anyway — "all layers loaded, camera at z=32 in a 12-deep volume" is a diagnosis; a black screenshot isn't.

**Tag objects to tell "updated" from "rebuilt".** Stamping `managed.__mvsTag` on live layers and re-reading it after an update was the only way to prove shaders and layout survived — and it's what confirmed the layer-diffing work.

**Ask the API instead of guessing.** For the contrast问题 I called `trackable.restoreState(...)` with three candidate shapes and printed the thrown errors. That showed in one run that `{range, window}` is accepted and adding `channel` breaks it — after I'd already guessed wrong once.

## Reproduce the deployment, not a convenient approximation

Two bugs were invisible until the layout matched production:

- Serving the page at `/browser/` with the bundle at `/browser/neuroglancer/` exposed the `../async_computation.bundle.js` 404 — the missing decode worker behind "OME-Zarr shows up grey".
- Serving the "CDN" from a **second origin and counting requests** proved the service-worker cache, its single-flight join (4 workers → 1 fetch), and that cached responses still pass SRI.

**A test that can't fail is worse than none.** My first Pyodide-lockfile test wrote the lockfile next to the wheels, so the wrong `packageBaseUrl` resolved correctly by accident and it passed — then broke your app. Putting the lockfile in a directory of its own is now the whole point of that test, and it's stated as such in the file.

## Cheap layers around the expensive one

- **Python-side end-to-end serving** (walk every key, decode every chunk) cleanly separated "Python is wrong" from "the browser is wrong" — it's what proved the fused preview was healthy before the search moved to `viewer.js`.
- **Pure logic extracted to `camera.js`** so the camera rules run under `node --test` with no DOM.
- **Source-level assertions** in `routes.test.mjs` as guards for behaviour only a browser can exercise — e.g. "`lockFileURL` is never passed without `packageBaseUrl`", "`setLayerTransforms` must not look layers up by name".
- **Faithful environment reproduction** for the CI failure: pytest 7.4.4 on 3.13 with the user variables stripped and `pwd` blocked reproduced the exact `OSError` message before I touched `tox.ini`.

## The recurring shape

Nearly every bug was a **platform boundary**, not an algorithm: URL resolution, cache identity, dimension ordering, when a shader gets generated. They all failed silently — a blank viewer, a grey layer, a reset layout — with Python entirely correct. The tests that mattered were the ones that put the two sides together in the arrangement they actually ship in.

Honest limit: none of this verifies rendered pixels. Everything above is one layer short of what you see, which is why your round-trips stayed necessary.