"""Load and preprocess raw ERA5 reanalysis data.
"""

import glob
import json
from itertools import chain
from datetime import datetime, timedelta

import dask
import numpy as np
import xarray as xr
import netCDF4 as nc


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
                samples_per_npz: int=28,
                plevels: list=None,
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
                                 preprocess=_convert_to_time_idx,
                                 concat_dim='time',
                                 data_vars='minimal',
                                 coords='minimal',
                                 compat='override',
                                 engine='h5netcdf',
                                 parallel=parallel)

    # dset = _process_dataset(dset, var.toupper(), time_step, plevels))

    return dset


def _convert_to_time_idx(dset):
    """This should only be used on datasets that fit in memory.
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
            var: xr.concat([tup[1] for tup in data], dim=times),
            # 'utc_date': utc_date
        }).drop(['forecast_initial_time', 'forecast_hour'])

    assert 'time' in dset.variables, \
        f'Could not create a time index for dataset from file {dset.encoding["source"]}'

    return dset

def _find_primary_var_name(dset):
    multi_dim_vars = [var.name for var in dset.data_vars.values() if len(var.dims) > 1]
    assert len(multi_dim_vars) == 1, \
        f'Don\'t know how to find primary var name for dataset from file {dset.encoding["source"]}'
    return multi_dim_vars[0]


def _process_dataset(dset, time_step, plevels, var=None):
    """Extract samples points from dset, returning an xr.DataArray.
    """
    if var is None:
        var = _find_primary_var_name(dset)

    if 'level' in dset.variables: # plevels data

        pl_idx = np.where(np.in1d(dset['level'][:], plevels))[0]

        # time indices for subsetting samples from the dataset
        time_idx = np.arange(0, dset['time'].size, time_step)

        darray = dset[var][time_idx, pl_idx, :, :]

    elif 'forecast_initial_time' in dset.variables: # aggregated data

        # first time in dset
        t_0 = dset['forecast_initial_time'][0].data.astype('datetime64[h]')

        # distance between forecast initial times
        fi_delta = dset['forecast_hour'][-1].data.astype('timedelta64[h]')

        # the last time in dataset
        last_t = dset['forecast_initial_time'][-1].data + fi_delta

        # times for unused sample points in the dataset
        time_idx = np.arange(t_0, last_t, np.timedelta64(time_step, 'h'), dtype='datetime64[h]')

        # indices of forecast initial time for the samples
        fi_idx = ((time_idx - t_0) // fi_delta)# .astype('datetime64[h]')

        # indices of forecast hour for the samples
        fh_idx = (time_idx - t_0) % fi_delta

        darray = dset[var][fi_idx, fh_idx, :, :]

    return darray
