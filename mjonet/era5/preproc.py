"""Load and preprocess raw ERA5 reanalysis data.
"""

import re
import time

from itertools import chain

import dask
import numpy as np
import xarray as xr

from .util import make_regridder


__all__ = ['preprocess',
           'preprocess_univariate',
           'subsample']


def preprocess(paths,
               time_step: int=6,
               plevels=None,
               resolution: float=0.25,
               regridder_weights_dir=None,
               chunks: dict=None,
               parallel: bool=False,
               verbose: bool=False):
    """Read in ERA5 data stored in HDF5 format, extract specified variables at specified time
    intervals and pressure levels, regrid to given resolution, and return the xr.Dataset.

    Parameters
    __________
    paths: Union[List[str], Dict[str, List[str]]]
        File paths to data, as returned by `get_filepaths`.
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
        See `glob_dict` parameter in `get_filepaths`.
    year_or_range: Tuple[int], optional
        See `year_or_range` parameter in `get_filepaths`.
    era5_dir: str, optional
        See `era5_dir` parameter in `get_filepaths`.
    regridder_weights_dir: str, optional
        If provided, used to look for existing or save new regridding weights.
    chunks: dict, optional
        A mapping from dimension names to chunk sizes.
    parallel: bool, optional
        If True (the default), open and preprocess files in parallel via `dask.delayed`. Passed as
        parameter of the same name to `xarray.open_mfdataset`.
    verbose: bool, optional
        If True, print progress.
    """
    tic = time.perf_counter()
    if isinstance(paths, dict):
        paths = list(chain(*paths.values()))

    if verbose:
        print(f'Preprocessing {len(paths)} files...')

    with dask.config.set({'array.slicing.split_large_chunks': False}, config=dask.config.config):
        # TODO: add xr_kwargs parameter
        dset = xr.open_mfdataset(paths,
                                 preprocess=preprocess_univariate,
                                 data_vars='minimal',
                                 coords='minimal',
                                 compat='override',
                                 engine='h5netcdf',
                                 chunks=chunks,
                                 parallel=parallel)

        if verbose:
            print(f'Datasets opened and combined. Removing missing times...')

        dset = _remove_missing_times(dset, paths)

        if verbose:
            print(f'Missing times removed. Subsampling and regridding...')

        dset.encoding['source'] = 'mjonet.data.era5.process'
        dset = subsample(dset,
                         time_step,
                         plevels)

        dset = dset.rename({'latitude': 'lat', 'longitude': 'lon'})

        if resolution != 0.25:
            regridder = make_regridder(dset,
                                       resolution,
                                       regridder_weights_dir)
            schema = dset.to_dict(data=False)
            dset = regridder(dset)
            dset.attrs.update({ 'schema_before_regrid': schema })

        if chunks is not None:
            dset = dset.chunk(chunks)

    if verbose:
        print(f'Finished preprocessing | time elapsed: {time.perf_counter() - tic:.2f}s\n')

    return dset


def preprocess_univariate(dset):
    """Preprocess a dataset with a single ERA5 variable by creating 'time' from
    'forecast_initial_time' and 'forecast_hour', if necessary.
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


def subsample(dset,
              time_step=None,
              plevels=None):
    """Create a new `xarray.Dataset` by extracting samples points and pressure levels from `dset`.

    Parameters
    __________
    dset: xarray.Dataset
        The dataset to subsample and regrid.
    time_step: int, optional
        See `time_step` parameter in `preprocess`.
    plevels: Union[List[int], Dict[str, List[int]]], optional
        See `plevels` parameter in `preprocess`. If not None, new variables are created for each
        level with names like {var_name}_{level} and `level` dimension is dropped.

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
        dset = dset.drop_dims('level')

    elif plevels is not None:
        raise ValueError('plevels should be a list or a dict')

    # dset = dset.reindex(lat=dset.lat[::-1])
    return dset


def _remove_missing_times(dset, paths):
    """Remove times with completely missing data from `dset`, based on file paths used to create the
    dataset.
    """
    var_ranges = {}

    for path in paths:

        match = re.search(r'\.[0-9]{3}_[0-9]{3}_([a-z0-9]*)\.', path)
        var_name = match[1]

        file_start, file_end = _timerange_from_path(path)

        var_start, var_end = None, None

        # update var_ranges
        if var_name in var_ranges.keys():
            var_start, var_end = var_ranges[var_name]

        # update lower bound
        if var_start is None or file_start < var_start:
            var_start = file_start

        # update upper bound
        if var_end is None or file_end > var_end:
            var_end = file_end

        var_ranges[var_name] = (var_start, var_end)

    # find max lower bound and min upper bound
    lower = np.amax([var_range[0] for var_range in var_ranges.values()])
    upper = np.amin([var_range[1] for var_range in var_ranges.values()])

    keep = np.logical_and(dset['time'].data >= lower, dset['time'].data <= upper)
    indexers = { 'time': dset['time'].data[keep] }

    return dset.sel(indexers)


def _timerange_from_path(path):
    """Get the range of dates from an ERA5 file path and return as tuple of datetimes.
    """
    match = re.search(r'\.([0-9]{10})_([0-9]{10})\.nc$', path)
    assert match and match.lastindex == 2, f'The file path has unexpected format: {path}'

    start, end = match[1], match[2]

    file_start = np.datetime64(
        f'{start[:4]}-{start[4:6]}-{start[6:8]}T{start[8:]}'
    ).astype('datetime64[ns]')

    file_end = np.datetime64(
        f'{end[:4]}-{end[4:6]}-{end[6:8]}T{end[8:]}'
    ).astype('datetime64[ns]')

    # check if the file is for a forecast
    if 'e5.oper.fc' in path:
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
