"""Load and preprocess raw ERA5 reanalysis data.
"""

import re
import glob
import json
from datetime import datetime, timedelta

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
    save_dir = save_dir[:-1] if save_dir[-1] == '/' else save_dir

    var_list = list(var_names)
    var_list.sort()

    if plevels is None:
        plevels = [200, 500, 850]

    filename_dict = get_filenames(*var_list, glob_dict=glob_dict,
                                  year_or_range=year_or_range, era5_dir=era5_dir)

    # get the start and end datetimes (inclusive) for the first sequence of samples
    first_file = filename_dict[var_list[0]][0]
    samples_start = _daterange_from_filename(first_file)[0]
    samples_end = samples_start + timedelta(hours=time_step*(samples_per_npz - 1))

    open_data = {} # dict for in-use NetCDF4 Datasets
    var_dim_len = 0 # var dimension, including levels
    for var in var_list:

        # open the first var file to examine
        dset = nc.Dataset(filename_dict[var][0], 'r')

        # get the idx in the var dimension for this variable
        if 'level' in dset.variables:
            var_idx = list(range(var_dim_len, var_dim_len + len(plevels)))
            var_dim_len += len(plevels)
        else:
            var_idx = var_dim_len
            var_dim_len += 1

        dset.close()

        # create metadata dict for var
        open_data[var] = {
            'var': var.upper(),
            'var_idx': var_idx, # position of var levels in sample array
            'dset': None,
            'dset_end': None,
            'dset_idx': None, # first unused sample for a new batch of data
            'samples_idx': 0 # next index to process in sequence of grouped samples
        }

    # variables x levels, latitude (height), longitude (width)
    dset = nc.Dataset(first_file, 'r')
    sample_shape = (var_dim_len, dset['latitude'].size, dset['longitude'].size)
    dset.close()

    n_written = 0
    while samples_start is not None:

        # create a new set of samples that will be saved together
        samples = {
            'samples': [np.empty(sample_shape, dtype=np.float32) for _ in range(samples_per_npz)],
            'samples_n': samples_per_npz,
            'samples_end': samples_end
        }

        # loop over the vars
        for var in var_list:

            # check for already opened data
            if open_data[var]['dset'] is not None:

                # there's a cached dset with samples we need for the new sequence
                # happens when time_step * samples_per_npz is shorter than time range of files

                # extract samples
                _process_open_dataset(open_data[var], samples, time_step, plevels)

            if open_data[var]['samples_idx'] < samples_per_npz:

                # there was no cached data or it didn't have enough samples to
                # complete the sample sequence, so create a new multi-file
                # dataset (xr.Dataset)

                dset_end = datetime(1900, 1, 1, 0)
                dset_files = []

                # check current end time and whether there are any files left
                while dset_end < samples_end and len(filename_dict[var]) > 0:
                    # get the next file, update end time, and add to file list
                    next_file = filename_dict[var].pop(0)
                    dset_range = _daterange_from_filename(next_file)
                    dset_end = dset_range[1]
                    dset_files.append(next_file)

                # if needed, add more files for the final sample's target
                fname_iter = iter(filename_dict[var])
                fname = next(fname_iter, None)
                while dset_end < samples_end and fname is not None:
                    dset_end = _daterange_from_filename(fname)[1]
                    dset_files.append(fname)
                    fname = next(fname_iter, None)

                # update samples_per_npz and samples_end using the actual number we got
                if dset_end < samples_end:
                    samples_per_npz = (dset_end - samples_start) // timedelta(hours = time_step)
                    samples_end = samples_start + timedelta(hours=time_step*samples_per_npz)

                # create a multi-file xr.Dataset
                open_data[var]['dset'] = xr.open_mfdataset(dset_files)
                open_data[var]['dset_end'] = dset_end

                # extract samples
                _process_open_dataset(open_data[var], samples, time_step, plevels)

            # reset index in samples
            open_data[var]['samples_idx'] = 0

        stage = f'-{stage}' if stage else ''
        file_prefix = f'{save_dir}/era5{stage}-{str(n_written).zfill(9)}'
        np.save_z_compressed(f'{file_prefix}.npz', *samples['samples'])
        metadata = {
            'var_idx': { var: open_data[var]['var_idx'] for var in var_list },
            'plevels': plevels,
            'time_step': time_step,
            'n_samples': samples_per_npz,
            'start': str(samples_start),
            'end': str(samples_end)
        }
        with open(f'{file_prefix}.json', 'w', encoding='utf-8') as outfile:
            json.dump(metadata, outfile, ensure_ascii=False, indent=4)
        n_written += 1
        if verbose:
            print(f'Wrote {file_prefix}' + '.{npz,json}')

        # check if there are more samples to process
        if np.all([open_data[var]['dset'] is not None or len(filename_dict[var]) > 0
                   for var in var_list]):
            samples_start = samples_end + timedelta(hours=time_step)
            samples_end = samples_start + timedelta(hours=time_step*(samples_per_npz - 1))

        else:
            # if not, we're finished
            samples_start = None
    # clean up
    for var in var_list:
        if open_data[var]['dset'] is not None:
            open_data[var]['dset'].close()


def _daterange_from_filename(filename):
    """Get the range of dates from an ERA5 filename and return as tuple of datetimes.
    """
    match = re.search(r'\.([0-9]{10})_([0-9]{10})\.nc$', filename)
    assert match and match.lastindex == 2, f'The filename has unexpected format: {filename}'
    start, end = match[1], match[2]
    return (datetime(year=int(start[:4]), month=int(start[4:6]),
                     day=int(start[6:8]), hour=int(start[8:])),
            datetime(year=int(end[:4]), month=int(end[4:6]),
                     day=int(end[6:8]), hour=int(end[8:])))


def _process_open_dataset(dataset_dict, samples_dict, time_step, plevels):

    var = dataset_dict['var']
    var_idx = dataset_dict['var_idx']

    dset = dataset_dict['dset']
    dset_idx = dataset_dict['dset_idx']
    dset_end = dataset_dict['dset_end']

    samples = samples_dict['samples']
    samples_n = samples_dict['samples_n']

    close_dset = dset_end < samples_dict['samples_end'] + timedelta(hours=time_step)

    if 'level' in dset.variables: # plevels data

        pl_idx = np.where(np.in1d(dset['level'][:], plevels))[0]

        # the final index of 'time' to get from the dataset
        if dset_idx is None:
            dset_idx = 0
            stop_idx = dset['time'].size
        else:
            stop_idx = min(dset_idx + samples_n*time_step, dset['time'].size)

        # time indices for subsetting samples from the dataset
        time_idx = list(range(dset_idx, stop_idx, time_step))

        for j, t_idx in enumerate(time_idx):
            # add data to the sample
            samples[j][var_idx, :, :] = dset[var][t_idx, pl_idx, :, :]

        if len(time_idx) < samples_n:
            # don't have all the samples yet, so remember where we left off
            dataset_dict['samples_idx'] = len(time_idx)

        # dataset may have additional unused sample points
        dataset_dict['dset_idx'] = time_idx[-1] + time_step

        if close_dset:
            # update index to first sample position in next dataset
            dataset_dict['dset_idx'] = dataset_dict['dset_idx'] % dset['time'].size

    elif 'forecast_initial_time' in dset.variables: # aggregated data

        if dset_idx is None:
            fi_idx = 0 # forecast_initial_time (aka forecast_reference_time)
            fh_idx = 0 # forecast_hour (aka forecast_period)
        else:
            fi_idx = dset_idx[0]
            fh_idx = dset_idx[1]

        # reference time
        fi_time = dset['forecast_initial_time'][fi_idx]

        # distance between forecast initial times
        fh_size = dset['forecast_hour'].size

        # the last time in dataset
        last_time = dset['forecast_initial_time'][-1] + fh_size - 1

        # times for unused sample points in the dataset
        unused_times = [dset['forecast_initial_time'][fi_idx] + fh_idx]
        next_time = unused_times[-1] + time_step

        # add sample point times until we run out or don't need more
        while next_time <= last_time and len(unused_times) < samples_n:
            unused_times.append(next_time)
            next_time += time_step

        # indices of forecast initial time for the samples
        fi_idxs = [fi_idx + (t - fi_time) // fh_size for t in unused_times]

        # indices of forecast hour for the samples
        fh_idxs = [(t - fi_time) % fh_size for t in unused_times]

        for j in range(len(unused_times)):
            # add data to the sample
            samples[j][var_idx, :, :] = dset[var][fi_idxs[j], fh_idxs[j], :, :]

        if len(unused_times) < samples_n:
            # don't have all the samples yet, need to remember where we left off
            dataset_dict['samples_idx'] = len(unused_times)

        # dataset may have additional unused sample points
        next_time += time_step
        next_fi_idx = fi_idx + (next_time - fi_time) // fh_size
        next_fh_idx = (next_time - fi_time) % fh_size

        if close_dset:
            dataset_dict['dset_idx'] = (
                next_fi_idx % dset['forecast_initial_time'].size,
                next_fh_idx
            )
        else:
            dataset_dict['dset_idx'] = (next_fi_idx, next_fh_idx)

    # check if we used up the dataset
    if close_dset:
        # done with open dataset, close it
        dataset_dict['dset'].close()
        dataset_dict['dset'] = None
        dataset_dict['dset_end'] = None
