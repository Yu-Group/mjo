# Interpretable forecasting of the Madden-Julian Oscillation 

## Notebooks / example scripts

- `00-preproc` -- Preprocessing examples for the [ERA5](https://confluence.ecmwf.int/display/CKB/ERA5) reanalysis data repository on [NERSC Cori and Perlmutter systems](https://docs.nersc.gov/systems/):
    - `00-nc-intro.ipynb`: Basics of ERA5's netCDF4 data model.
    - `01-cartopy-viz.ipynb`: Visualization of ERA5 data using [`cartopy`](https://scitools.org.uk/cartopy/docs/latest/).
    - `02-data-preprocessing.ipynb`: ERA5 data preprocessing examples.
    - `03-webdataset.ipynb`: Examples for creating an ERA5 [`WebDataset`](https://pytorch.org/blog/efficient-pytorch-io-library-for-large-datasets-many-files-many-gpus/).
    - `04-preproc-fourcastnet.py`: Script to create full resolution (0.25) training, validation, and test ERA5 `WebDatasets`.
    - `env.yml`: Specification to create `mjonet-preproc` conda environment for running these examples.

## Installation

### Local development

```py
python setup.py develop
```

### From GitHub

```py
python -m pip install git+git://github.com/Yu-Group/mjo@main
```

### conda

```sh
# create conda environment and Jupyter kernel
ENV=mjonet-preproc
ENV_PATH=notebooks/00-preproc/env.yml

conda create -f $ENV_PATH
conda activate $ENV
python setup.py develop
python -m ipykernel install --user --name=$ENV --display-name="Python [conda:$ENV]"
```
