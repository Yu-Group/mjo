"""Load and preprocess raw ERA5 reanalysis data.
"""

import glob
import json
from itertools import chain

import dask
import numpy as np
import xarray as xr
import xesmf as xe


def preprocess(*var_names: str,
               time_step: int=6,
               plevels: list=None,
               resolution: float=0.25,
               glob_dict: dict=None,
               year_or_range: tuple=None,
               era5_dir: str='/global/cfs/projectdirs/m3522/cmip6/ERA5',
               save_dir: str=None,
               steps_per_chunk: int=None,
               stage: str=None,
               parallel: str=True,
               verbose: bool=False):
    """Read in ERA5 data stored in HDF5 format, extract specified variables at specified time
    intervals and pressure levels, regrid to a given resolution, and save a specified number of
    data samples in a single compressed .zarr file named like f'era5-{stage}-000000.zarr', together
    with a corresponding JSON metadata file.

    Parameters
    __________
    var_names: List[str]
        See `var_names` parameter in `get_filenames`.
    time_step: int, optional
        The frequency at which to sample points in the time domain. The real-time length depends on
        the resolution of the data, but we assume 1 step = 1 hour for ERA5. Default frequency is
        every 6 time points.
    plevels: Union[List[int], Dict[str, List[int]]], optional
        The specific values at which to sample points in the pressure level domain. Units are hPa.
        Either a list of levels, in which case the same levels are used for every variable that has
        levels, or a dictionary where the key is a variable names and the value is lists of levels
        for that variable. If `None`, all levels are included.
    resolution: float, optional
        The vertical / horizontal spacing between grid points, in degrees. The default 0.25 is the
        original resolution of the ERA5 dataset.
    glob_dict: Dict[str, str], optional
        See `glob_dict` parameter in `get_filenames`.
    year_or_range: Tuple[int], optional
        See `year_or_range` parameter in `get_filenames`.
    era5_dir: str, optional
        See `era5_dir` parameter in `get_filenames`.
    save_dir: str, optional
        If provided, saves outputs in the directory with filenames like f'era5-{stage}-000000.zarr'.
    steps_per_chunk: int, optional
        Number of time points to include in each chunk of `xarray.DataArray`s and output files. If
        `None`, uses the chunk size set by dask.
    stage: str, optional
        If provided, should be one of 'train', 'validate', and 'test'.
    parallel: bool, optional
        If True (the default), open and preprocess files in parallel via `dask.delayed`. Passed as
        parameter of the same name to `xarray.open_mfdataset`.
    verbose: bool, optional
        If True, print progress.
    """
    save_dir = save_dir[:-1] if save_dir[-1] == '/' else save_dir

    var_list = list(var_names)
    var_list.sort()

    fnames = get_filenames(*var_list, glob_dict=glob_dict,
                           year_or_range=year_or_range, era5_dir=era5_dir)
    fnames = list(chain(*fnames.values()))

    if verbose:
        print(f'Processing {len(fnames)} files.')

    chunks = None
    if steps_per_chunk is not None:
        chunks = { 'time': steps_per_chunk }

    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        dset = xr.open_mfdataset(fnames,
                                 preprocess=_preprocess_one_file,
                                 data_vars='minimal',
                                 coords='minimal',
                                 compat='override',
                                 engine='h5netcdf',
                                 parallel=parallel)

    if verbose:
        print(f'Datasets opened and combined. Subsampling and regridding.')

    dset.encoding['source'] = 'mjonet.data.era5.process'

    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        dset = subsample_and_regrid(dset, time_step, plevels, resolution)

    return dset


def subsample_and_regrid(dset,
                         time_step=6,
                         plevels=None,
                         resolution=0.25,
                         rename_lat_lon=True):
    """Extract samples points from `dset` and regrid the data.

    Parameters
    __________
    dset: xarray.Dataset
        The dataset to subsample and regrid.
    time_step: int, optional
        See `time_step` parameter in `preprocess`.
    plevels: Union[List[int], Dict[str, List[int]]], optional
        See `plevels` parameter in `preprocess`.
    resolution: float, optional
        See `resolution` parameter in `preprocess`.
    rename_lat_lon: bool, optional
        If True (the default), rename 'latitude' to 'lat' and 'longitude' to 'lon'.
    """        
    # subsample time points
    dset = dset.isel({'time': np.arange(0, dset['time'].size, time_step)})

    # subsample pressure levels
    indexers = {}

    if isinstance(plevels, list):
        var_names = _find_primary_var_names(dset)
        for var in var_names:
            indexers[f'{var.lower()}_level'] = levels
    elif isinstance(plevels, dict):
        for var, levels in plevels.items():
            indexers[f'{var.lower()}_level'] = levels
    else:
        raise ValueError('plevels should be a list or a dict')

    dset = dset.sel(indexers)

    if rename_lat_lon:
        assert 'latitude' in dset.variables and 'longitude' in dset.variables, \
            'Can\'t rename "latitude" and "longitude" variables because one or both are missing'
        dset = dset.rename({'latitude': 'lat', 'longitude': 'lon'})

    # regrid
    if resolution > 0.25:
        grid = _grid_era5(dset, resolution)
        regridder = xe.Regridder(dset, grid, 'conservative', periodic=True)
        dset = regridder(dset)

    return dset


def get_filenames(*var_names: str,
                  glob_dict: dict=None,
                  year_or_range: tuple=None,
                  era5_dir: str='/global/cfs/projectdirs/m3522/cmip6/ERA5'):

    """Get filenames for each input variable. Default settings make many assumptions about the
    directory/filename structure.

    Parameters
    __________
    var_names: List[str]
        The variable names ("Short Name"), as specified by https://rda.ucar.edu/datasets/ds633.0/.
    glob_dict: Dict[str, str], optional
        A dictionary with entries like (var_name: glob_str) where values are glob strings to be
        passed to glob.glob() to find the necessary files.
    year_or_range: Union[int, Tuple[int]], optional
        A single year or a 2-tuple where the first element is the starting year and the second is
        the ending year (inclusive).
    era5_dir: str, optional
        The directory of the CMIP6 ERA5 datasets (no trailing slash). Default is the location on
        Perlmutter.

    Returns
    _______
    filename_dict: dict
        A dictionary with entries like (var_name: filename_list)

    """
    era5_dir = era5_dir[:-1] if era5_dir[-1] == '/' else era5_dir
    if year_or_range is None:
        year_glob = '*'
    elif isinstance(year_or_range, tuple):
        year_glob = '{' + str(year_or_range[0]) + '..' + str(year_or_range[1]) + '}[0-1][0-9]'
    elif isinstance(year_or_range, int):
        year_glob = f'{year_or_range}[0-1][0-9]'
    if glob_dict is None:
        pl_dir = f'{era5_dir}/e5.oper.an.pl'
        glob_dict = {
            'z': f'{pl_dir}/{year_glob}/e5.oper.an.pl.128_129_z.ll025sc.*.nc',
            't': f'{pl_dir}/{year_glob}/e5.oper.an.pl.128_130_t.ll025sc.*.nc',
            'u': f'{pl_dir}/{year_glob}/e5.oper.an.pl.128_131_u.ll025uv.*.nc',
            'v': f'{pl_dir}/{year_glob}/e5.oper.an.pl.128_132_v.ll025uv.*.nc',
            'r': f'{pl_dir}/{year_glob}/e5.oper.an.pl.128_157_r.ll025sc.*.nc',
            'mtnlwrf': (f'{era5_dir}/e5.oper.fc.sfc.meanflux/{year_glob}/' \
                         'e5.oper.fc.sfc.meanflux.235_040_mtnlwrf.ll025sc.*.nc')
        }
    filename_dict = {}
    var_list = list(var_names)
    var_list.sort()
    for var in var_list:
        filename_dict[var] = glob.glob(glob_dict[var])
        filename_dict[var].sort()
    return filename_dict


def _find_primary_var_names(dset):
    multi_dim_vars = [var.name for var in dset.data_vars.values() if len(var.dims) > 1]
    assert len(multi_dim_vars) == 1, \
        f'Don\'t know how to find primary var name for dataset from {dset.encoding["source"]}'
    if len(multi_dim_vars) == 1:
        return multi_dim_vars[0]
    return multi_dim_vars


def _preprocess_one_file(dset):
    """Preprocess a dataset from a single ERA5 file by renaming 'level' and creating 'time' from
    'forecast_initial_time' and 'forecast_hour'.
    """
    # try to get the main variable name
    var = _find_primary_var_names(dset)

    if 'level' in dset.variables:
        dset = dset.rename({'level': f'{var.lower()}_level'})

    if 'forecast_initial_time' in dset.variables:

        # extract time x (lat, lon)
        data = [
            (init.data + hour.data.astype('timedelta64[h]'),
             dset[var].sel(forecast_initial_time = init, forecast_hour = hour))
            for init in dset['forecast_initial_time']
            for hour in dset['forecast_hour']
        ]

        # done with original dataset, close it
        dset.close()

        # make the time dimension and utc_date var
        times = [tup[0] for tup in data]
        times = xr.DataArray(times, coords={'time': times},
                             dims=('time',), name='time',
                             attrs={'long_name': 'time'})

        # create the new dataset indexed by time
        dset = xr.Dataset({
            var: xr.concat([tup[1] for tup in data], dim=times)
        }).drop(['forecast_initial_time', 'forecast_hour'])

    assert 'time' in dset.variables, \
        f'Could not create a time index for dataset from {dset.encoding["source"]}'

    return dset


def _grid_era5(dset, res):
    """Like xesmf.util.grid_2d, but keeping ERA5 data conventions. See
    https://github.com/JiaweiZhuang/xESMF/blob/master/xesmf/util.py.
    """
    # bounds
    # latitude ranges from high to low in ERA5
    lat_b = np.arange(dset['lat'].max(), dset['lat'].min() - res, -res)
    lon_b = np.arange(dset['lon'].min(), dset['lon'].max() + res, res)

    # centers
    lat = (lat_b[:-1] + lat_b[1:])/2
    lon = (lon_b[:-1] + lon_b[1:])/2

    grid = xr.Dataset(coords={
        'lon': (['lon'], lon),
        'lat': (['lat'], lat),
        'lon_b': (['lon_b'], lon_b),
        'lat_b': (['lat_b'], lat_b)
    })

    return grid
