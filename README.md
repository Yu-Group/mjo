# Interpretable forecasting of the Madden-Julian Oscillation 

List of notebooks / scripts:

- `00-preproc` -- Preprocessing examples for the [ERA5](https://confluence.ecmwf.int/display/CKB/ERA5) reanalysis data repository on [NERSC Cori and Perlmutter systems](https://docs.nersc.gov/systems/):
    - `00-nc-intro.ipynb`: Basics of ERA5's netCDF4 data model.
    - `01-cartopy-viz.ipynb`: Visualization of ERA5 data using [`cartopy`](https://scitools.org.uk/cartopy/docs/latest/).
    - `02-data-preprocessing.ipynb`: ERA5 data preprocessing examples.
    - `03-webdataset.ipynb`: Examples for creating an ERA5 [`WebDataset`](https://pytorch.org/blog/efficient-pytorch-io-library-for-large-datasets-many-files-many-gpus/).
    - `04-preproc-fourcastnet.py`: Script to create full resolution (0.25) training, validation, and test ERA5 `WebDatasets`.
