"""Load and preprocess raw ERA5 reanalysis data.
"""

import io
import gc
import os
import re
import glob
import json

# TOOD: remove
import time

from datetime import datetime, timedelta
from itertools import chain

import dask
import numpy as np
import xesmf as xe
import xarray as xr
import webdataset as wds


__all__ = ['create_wds', 
           'get_filenames', 
           'preprocess', 
           'subsample_and_regrid']


def create_wds(*var_names: str,
               save_dir: str,
               samples_per_tar: int,
               time_step: int=6,
               target_steps: list=None,
               plevels: list=None,
               resolution: float=0.25,
               glob_dict: dict=None,
               year_or_range: tuple=None,
               era5_dir: str='/global/cfs/projectdirs/m3522/cmip6/ERA5',
               stage: str=None,
               parallel: str=False,
               verbose: bool=False):
    """Create a WebDataset.
    
    Parameters
    __________
    dset: xarray.Dataset
        Dataset to base the WebDataset on.
    save_dir: str
        Saves outputs in the directory with filenames like f'era5-{stage}-000000.zarr'.
    samples_per_tar: int
        Number of (input, output) pairs to include in each tar file.
    var_names: List[str]
        The variable names ("Short Name"), as specified by https://rda.ucar.edu/datasets/ds633.0/.
    time_step: int, optional
        The frequency at which to sample points in the time domain. The real-time length depends on
        the temporal resolution of the data, but we assume 1 step = 1 hour for ERA5. Default 
        frequency is every 6 time points.
    target_steps: List[int], optional
        The temporal step size from input observation to target observation. Should be a multiple
        of `time_step`, and there can be multiple targets. Default is same as `time_step`.
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
    stage: str, optional
        If provided, should be one of 'train', 'validate', and 'test'.
    parallel: bool, optional
        If True (the default), open and preprocess files in parallel via `dask.delayed`. Passed as
        parameter of the same name to `xarray.open_mfdataset`.
    verbose: bool, optional
        If True, print progress.
    """
    if target_steps is None:
        target_steps = [time_step]
        
    for t_step in target_steps:
        assert t_step % time_step == 0, 'Each of `target_steps` must be a multiple of `time_step`'    
    
    # convert target_steps to indices along dset 'time' axis
    target_steps = np.array(target_steps) // time_step

    # cushion needed at the end of the dataset
    steps_to_last_target = np.amax(target_steps) // time_step

    with preprocess(*var_names,
                    time_step=time_step,
                    plevels=plevels,
                    resolution=resolution,
                    glob_dict=glob_dict,
                    year_or_range=year_or_range,
                    era5_dir=era5_dir, 
                    chunks={'time': 1},
                    parallel=parallel,
                    verbose=verbose) as dset:

        times = dset['time'].data

        # loop over indices of first sample for each tar file
        indices = range(0, len(times) - steps_to_last_target, samples_per_tar)

        for tar_idx, start_idx in enumerate(indices):
            # create a tar file with `samples_per_tar` time point pairs from `dset`.

            tar_fname = f'era5-{stage}-{tar_idx:06d}.tar'
            tar_path = os.path.join(save_dir, tar_fname)

            if verbose:
                print(f'\rCreating tar file {tar_fname}...', end='')
                
            tic = time.perf_counter()

            # get the samples for this tar and persist in memory
            end_idx = min(start_idx + samples_per_tar + target_steps[-1], len(times))
            subset = dset.isel({ 'time': np.arange(start_idx, end_idx) }).compute()

            sink = wds.TarWriter(tar_path, encoder=False)

            # loop over the samples for this tar file
            for i in range(subset['time'].size - steps_to_last_target):

                x = subset.isel({'time': i})
                y = subset.isel({'time': i + target_steps})

                sink.write({
                    '__key__': f'sample{i:06d}',
                    'input.npz': _encode_era5(x),
                    'ouput.npz': _encode_era5(y)
                })

                x.close()
                y.close()

            subset.close()
            sink.close()

            if verbose:
                print(f'\rCreated tar file {tar_fname} ({tar_idx+1}/{len(indices)} : '
                      f'{100.*(tar_idx+1) / len(indices):.0f}%) | '
                      f'time taken: {time.perf_counter() - tic:.2f} s')


def _encode_era5(dset):
    """Save a small xr.Dataset to bytes in .npz format with entries for the ndarray and metadata
    dictionary, and return it as a string.
    """
    with io.BytesIO() as stream:
        data = {
            'ndarray': dset.to_array().to_numpy(),
            'metadata': dset.to_dict(data=False)
        }
        np.savez(stream, **data)

        return stream.getvalue()


def preprocess(*var_names: str,
               time_step: int=6,
               plevels: list=None,
               resolution: float=0.25,
               glob_dict: dict=None,
               year_or_range: tuple=None,
               era5_dir: str='/global/cfs/projectdirs/m3522/cmip6/ERA5',
               chunks=None,
               parallel: str=False,
               verbose: bool=False):
    """Read in ERA5 data stored in HDF5 format, extract specified variables at specified time
    intervals and pressure levels, regrid to a given resolution, and save a specified number of
    data samples in a single compressed .zarr file named like f'era5-{stage}-000000.zarr', together
    with a corresponding JSON metadata file.

    Parameters
    __________
    var_names: List[str]
        The variable names ("Short Name"), as specified by https://rda.ucar.edu/datasets/ds633.0/.
    time_step: int, optional
        The frequency at which to sample points in the time domain. The real-time length depends on
        the temporal resolution of the data, but we assume 1 step = 1 hour for ERA5. Default
        frequency is every 6 time points.
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
    parallel: bool, optional
        If True (the default), open and preprocess files in parallel via `dask.delayed`. Passed as
        parameter of the same name to `xarray.open_mfdataset`.
    verbose: bool, optional
        If True, print progress.
    """
    var_list = list(var_names)
    var_list.sort()

    fnames = get_filenames(*var_list, glob_dict=glob_dict,
                           year_or_range=year_or_range, era5_dir=era5_dir)
    flist = list(chain(*fnames.values()))

    if verbose:
        print(f'Processing {len(flist)} files...')
        
    with dask.config.set({'array.slicing.split_large_chunks': False}, config=dask.config.config):
        dset = xr.open_mfdataset(flist,
                                 preprocess=_preprocess_one_file,
                                 data_vars='minimal',
                                 coords='minimal',
                                 compat='override',
                                 engine='h5netcdf',
                                 chunks=chunks,
                                 parallel=parallel)

        if verbose:
            print(f'Datasets opened and combined. Removing missing times...')

        dset = _remove_missing_times(dset, fnames)

        if verbose:
            print(f'Missing times removed. Subsampling and regridding...')

        dset.encoding['source'] = 'mjonet.data.era5.process'
        dset = subsample_and_regrid(dset, time_step, plevels, resolution).chunk(chunks)

        if chunks is not None:
            dset = dset.chunk(chunks)

    return dset


def subsample_and_regrid(dset,
                         time_step=None,
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
    if time_step is not None:
        dset = dset.isel({'time': np.arange(0, dset['time'].size, time_step)})

    # create new variables by subsampling pressure levels
    
    if plevels is None and 'level' in dset:
        plevels = list(dset['level'].data)

    if isinstance(plevels, list):
        levels = plevels
        plevels = {}
        var_names = _find_primary_var_names(dset)
        for var in var_names:
            if 'level' in dset[var].dims:
                plevels[var] = levels

    if isinstance(plevels, dict):
        for var, levels in plevels.items():
            var = var.upper()
            for level in levels:
                # create a new variable from the var's data at level
                dset = dset.assign({f'{var}_{level}': dset[var].sel({'level': level})})
            dset = dset.drop_vars(var)
    
    elif plevels is not None:
        raise ValueError('plevels should be a list or a dict')

    if 'level' in dset.dims:
        dset = dset.drop_dims('level')    
    
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


def _remove_missing_times(dset, fnames):
    """Remove times with completely missing data from `dset`, based on filenames used to create the
    dataset.
    """
    var_ranges = {}

    for var, flist in fnames.items():

        var_start, var_end = None, None

        for fname in flist:

            file_start, file_end = _timerange_from_filename(fname)

            # update lower bound
            if var_start is None or file_start < var_start:
                var_start = file_start

            # update upper bound
            if var_end is None or file_end > var_end:
                var_end = file_end

        # update var_ranges
        var_ranges[var] = (var_start, var_end)

    # find max lower bound and min upper bound
    lower = np.amax([var_range[0] for var_range in var_ranges.values()])
    upper = np.amin([var_range[1] for var_range in var_ranges.values()])

    keep = np.logical_and(dset['time'].data >= lower, dset['time'].data <= upper)
    indexers = { 'time': dset['time'].data[keep] }

    return dset.sel(indexers)


def _timerange_from_filename(fname):
    """Get the range of dates from an ERA5 filename and return as tuple of datetimes.
    """
    match = re.search(r'\.([0-9]{10})_([0-9]{10})\.nc$', fname)
    assert match and match.lastindex == 2, f'The filename has unexpected format: {fname}'

    start, end = match[1], match[2]

    file_start = np.datetime64(
        f'{start[:4]}-{start[4:6]}-{start[6:8]}T{start[8:]}'
    ).astype('datetime64[ns]')

    file_end = np.datetime64(
        f'{end[:4]}-{end[4:6]}-{end[6:8]}T{end[8:]}'
    ).astype('datetime64[ns]')

    # check if the file is for a forecast
    if 'e5.oper.fc' in fname:
        # forecast file names start with earliest forecast_initial_time and end with latest
        # valid time (forecast_initial_time + forecast_hour)
        file_start += np.timedelta64(1, 'h')

    return file_start, file_end


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

    if 'forecast_initial_time' in dset.variables:

        # extract time x (lat, lon)
        data = [
            (init.data + hour.data.astype('timedelta64[h]'),
             dset[var].sel(forecast_initial_time = init, forecast_hour = hour))
            for init in dset['forecast_initial_time']
            for hour in dset['forecast_hour']
        ]

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

    if 'utc_date' in dset:
        dset = dset.drop_vars('utc_date')
    
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
