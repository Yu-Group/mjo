"""Load and preprocess raw ERA5 reanalysis data.
"""

import glob
import json
from itertools import chain

import dask
import numpy as np
import xarray as xr
import xesmf as xe

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


def hdf5_to_npz(save_dir,
                *var_names: str,
                time_step: int=6,
                plevels: list=None,
                resolution: float=0.25,
                samples_per_npz: int=28,
                glob_dict: dict=None,
                year_or_range: tuple=None,
                era5_dir: str='/global/cfs/projectdirs/m3522/cmip6/ERA5',
                stage: str=None,
                verbose: bool=False):

    """Read in ERA5 data stored in HDF5 format, extract specified variables at specified time
    intervals and pressure levels, stack variables into a single np.array for each time point,
    and save a specified number of data samples in a single compressed .npz file named like
    f'era5-{stage}-000000.npz', together with a corresponding JSON metadata file.

    Parameters
    __________
    save_dir: str
        The directory in which to save outputs. A the .npz files.
    var_names: List[str]
        See `var_names` parameter in `get_filenames`.
    time_step: int, optional
        The frequency at which to sample points in the time domain. The real-time length depends on
        the resolution of the data, but we assume 1 step = 1 hour for ERA5. Default frequency is
        every 6 time points.
    resolution: float, optional
        The vertical / horizontal spacing between grid points, in degrees. The default 0.25 is the
        original resolution of the ERA5 dataset.
    samples_per_npz: int, optional
        The number of data samples to write to a single output npz file. With time_step 6, the
        default of 28 gives a week of data per npz file.
    plevels: List[int]
        The specific values at which to sample points in the pressure level domain. Units are hPa.
        This is only used for a given `netCDF4.Dataset` if it has a variable called 'level'.
    glob_dict: Dict[str, str], optional
        See `glob_dict` parameter in `get_filenames`.
    year_or_range: Tuple[int], optional
        See `year_or_range` parameter in `get_filenames`.
    overwrite: bool, optional
        If True, overwrite any existing files in `save_dir`. Default is False, in which case
        existing files trigger a warning and nothing is written.
    era5_dir: str
        See `era5_dir` parameter in `get_filenames`.
    stage: str, optional
        If provided, should be one of 'train', 'validate', and 'test'.
    verbose: bool, optional
        If True, print progress.
    """

    parallel = False

    save_dir = save_dir[:-1] if save_dir[-1] == '/' else save_dir

    var_list = list(var_names)
    var_list.sort()

    if plevels is None:
        plevels = [200, 500, 850]

    fnames = get_filenames(*var_list, glob_dict=glob_dict,
                           year_or_range=year_or_range, era5_dir=era5_dir)

    fnames = list(chain(*fnames.values()))

    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        dset = xr.open_mfdataset(fnames,
                                 preprocess=_convert_to_time_index,
                                 # concat_dim='time',
                                 data_vars='minimal',
                                 coords='minimal',
                                 compat='override',
                                 engine='h5netcdf',
                                 parallel=parallel)

    dset.encoding['source'] = 'hdf5_to_npz'
    dset = _process_dataset(dset, var_list, time_step, plevels, resolution)

    return dset


def _convert_to_time_index(dset):
    """Convert initial time / hour time to a time index.
    """
    if 'time' in dset.variables:
        return dset

    if 'forecast_initial_time' in dset.variables:

        # try to get the main variable name
        var = _find_primary_var_name(dset)

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


def _find_primary_var_name(dset):
    # TODO: check if var.name is uppercase
    multi_dim_vars = [var.name for var in dset.data_vars.values()
                      if len(var.dims) > 1]
    assert len(multi_dim_vars) == 1, \
        f'Don\'t know how to find primary var name for dataset from {dset.encoding["source"]}'
    return multi_dim_vars[0]


def _process_dataset(dset, var_list, time_step, plevels, resolution):
    """Extract samples points from dset, returning an xr.DataArray.
    """

    indexers = {
        'time': np.arange(0, dset['time'].size, time_step),
        'level': [i for i, lev in enumerate(dset['level']) if lev in plevels]
    }

    # subsample the data
    dset = dset.isel(indexers).rename({'latitude': 'lat', 'longitude': 'lon'})

    # regrid
    if resolution > 0.25:
        grid = _grid_era5(dset, resolution)
        regridder = xe.Regridder(dset, grid, 'conservative', periodic=True)
        dset = regridder(dset)

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
