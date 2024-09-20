# Stitching in the browser

`multiview-stitcher` can run without installation in your browser.

## Try it out

- open [JupyterLite](https://jupyter.org/try-jupyter/lab/) in a private browser window
- upload this notebook into the jupyter lab window: [notebooks/stitching_in_the_browser.ipynb](https://github.com/multiview-stitcher/multiview-stitcher/tree/main/notebooks/stitching_in_the_browser.ipynb)
- upload files to stitch into a 'data' folder in the jupyter lab window
- follow the notebook

### Limitations
- stitching will run with a single thread
- while the code runs locally, your local file system is not directly accessible from within the browser environment

## This cool functionality is possible thanks to
- [JupyterLite](https://jupyter.org/try-jupyter/lab/), a JupyterLab distribution that runs in the browser
- [pyodide](https://pyodide.org/), a Python runtime for the browser
