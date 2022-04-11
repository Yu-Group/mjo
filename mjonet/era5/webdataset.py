"""Methods to create and load Webdatasets.
"""

import io
import os
import time
import pickle
import pathlib

import xesmf as xe
import numpy as np
import xarray as xr
import webdataset as wds

from .util import make_regridder
from .preproc import preprocess


__all__ = ['create_wds',
           'decode_npz']


def _encode(darray):
    """Save a small xr.DataArray to bytes in .npz format.
    dictionary, and return it as a string.
    """
    with io.BytesIO() as stream:
        data = {
            'data': darray.to_numpy(),
            'dims': darray.dims,
            'variable': darray['variable'].data,
            'time': darray['time'].data.astype('datetime64[h]').astype(np.int64),
            'lat': darray['lat'].data,
            'lon': darray['lon'].data,
        }
        np.savez(stream, **data)
        return stream.getvalue()


def _decode_npz(data):
    return np.load(io.BytesIO(data), allow_pickle=True)


decode_npz = wds.handle_extension("npz", _decode_npz)


def create_wds(fpaths,
               save_dir: str,
               samples_per_tar: int,
               stage: str=None,
               time_step: int=6,
               target_steps: list=None,
               plevels: list=None,
               resolution: float=0.25,
               regridder_weights_dir: str=None,
               chunks: dict=None,
               n_cached: int=None,
               parallel: str=False,
               overwrite: bool=False,
               verbose: bool=False):
    """Create a WebDataset by saving a specified number of data samples in a single tar file named
    like f'era5-{stage}-000000.tar'.

    Parameters
    __________
    paths: Union[List[str], Dict[str, List[str]]]
        File paths to data, as returned by `get_filepaths`.
    save_dir: str
        Save outputs in the `save_dir` directory, which should already exist. May also create a
        sub-directory in `save_dir`; see `stage` parameter.
    samples_per_tar: int
        Number of (input, output) pairs to include in each tar file.
    stage: str, optional
        If provided, create a subdirectory of `save_dir` with the same name and use `stage` in
        output filenames like f'era5-{stage}-000000.'. Common stage names are 'train', 'validate',
        and 'test'.
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
    regridder_weights_dir: str, optional
        See `regridder_weights_dir` parameter of `mjonet.era5.preprocess`.
    chunks: dict, optional
        See `chunks` parameter of `mjonet.era5.preprocess`.
    n_cached: int, optional
        The number of timepoints to cache in memory at one time. Together with the `resolution` and
        `chunks` parameters and the number of dask workers, this will effect the maximum memory used
        while running this function, along with its performance. If None, uses `samples_per_tar` +
        `np.amax(target_steps)`.
    parallel: bool, optional
        If True (the default), open and preprocess files in parallel via `dask.delayed`. Passed as
        parameter of the same name to `xarray.open_mfdataset`.
    overwrite: bool, optional
        If True, overwrites any files found in `save_dir` or `stage` subdirectory.
    verbose: bool, optional
        If True, print progress.

    """
    tic = time.perf_counter()

    if target_steps is None:
        target_steps = [time_step]

    for t_step in target_steps:
        assert t_step % time_step == 0, 'Each of `target_steps` must be a multiple of `time_step`'

    if stage is not None:
        save_dir = os.path.join(save_dir, stage)
        pathlib.Path(save_dir).mkdir(exist_ok=overwrite)

    if verbose:
        print(f'Saving WebDataset in {save_dir}...\n')

    # convert target_steps to sorted indices along dset 'time' axis
    target_steps = np.array(target_steps)
    target_steps.sort()
    target_steps = target_steps // time_step

    if n_cached is None:
        n_cached = samples_per_tar + target_steps[-1]

    regridder = None

    with preprocess(fpaths,
                    time_step=time_step,
                    plevels=plevels,
                    chunks=chunks,
                    parallel=parallel,
                    verbose=verbose) as dset:

        # save dset's schema
        with open(os.path.join(save_dir, 'schema.pkl'), 'wb') as f:
            pickle.dump(dset.to_dict(data=False), f)

        times = dset['time'].data

        # print(f'len(times): {len(times)}')

        # holds dset subsets for easy access
        # range_key: (subset, computed_bool)
        subsets = {
            range(i, i + n_cached): (
                dset.isel({ 'time': np.arange(i, min(i + n_cached, len(times))) }),
                False
            ) for i in range(0, len(times), n_cached)
        }

        # subset_index[i] gives key in subsets where we can find the computed
        # dset subset containing the timepoint with index i
        key_index = [key for key in subsets.keys() for _ in range(n_cached)]

        # loop over indices of first sample for each tar file
        start_indices = range(0, len(times) - target_steps[-1], samples_per_tar)

        for tar_idx, start_idx in enumerate(start_indices):

            if stage is None:
                tar_fname = f'era5-{tar_idx:06d}.tar'
            else:
                tar_fname = f'era5-{stage}-{tar_idx:06d}.tar'
            tar_path = os.path.join(save_dir, tar_fname)

            if verbose:
                print(f'\rCreating tar file {tar_fname}...', end='')

            # create tar file with `samples_per_tar` time point pairs from `dset`.

            toc = time.perf_counter()

            # tar's sample indices
            tar_indices = range(
                start_idx,
                min(start_idx + samples_per_tar, len(times) - target_steps[-1])
            )

            samples = []

            # loop over each sample in the tar
            for idx in tar_indices:

                if idx > 0 and key_index[idx] != key_index[idx - 1]:

                    # print(f'idx: {idx}')
                    # print(f'key_index[idx]: {key_index[idx]}')
                    # print(f'key_index[idx - 1]: {key_index[idx - 1]}')

                    # delete previous sample's computed subset which won't be used again
                    del subsets[key_index[idx - 1]]
                    # print(f'new compute: {[(k, v[1]) for k, v in subsets.items()]}\n')

                # indices in dset for timepoints in this sample
                time_idxs = [idx] + list(idx + target_steps)

                sample = []

                # loop over the timepoints in the sample
                for i in time_idxs:

                    # print(f'time index i: {i}')

                    # print(f'subset type: {type(subsets[key_index[i]])}')

                    subset, computed = subsets[key_index[i]]

                    if not computed:
                        # compute the subset
                        # print(f'Begin subset.compute()...')
                        # print(f'i: {i}')
                        # print(f'key_index[i]: {key_index[i]}')
                        # tic2 = time.perf_counter()
                        subset = subset.compute()
                        # print(f'End subset.compute(), time elapsed: {time.perf_counter() - tic2:.2f}s')
                        # print(f'subset.time range: {(subset.time.data[0], subset.time.data[-1])}\n')
                        subsets[key_index[i]] = (subset, True)

                    # print(f'times[i]: {times[i]}')
                    sample.append(subset.sel({ 'time': [times[i]] }))

                if regridder is None and resolution != 0.25:
                    # create / load the regridder
                    regridder = make_regridder(dset,
                                               resolution,
                                               regridder_weights_dir)

                if regridder is not None:
                    # use already instantiated regridder
                    sample = [regridder(obs) for obs in sample]

                # add sample to samples for this tar
                samples.append(sample)

            # create tar file

            sink = wds.TarWriter(tar_path, encoder=False)

            for i, sample in enumerate(samples):

                x = sample[0].to_array()
                y = xr.concat(sample[1:], dim='time').to_array()

                sink.write({
                    '__key__': f'{start_idx + i:06d}',
                    'input.npz': _encode(x),
                    'output.npz': _encode(y)
                })

            sink.close()

            if verbose:
                print(f'\rCreated tar file {tar_fname} ({tar_idx+1}/{len(start_indices)} : '
                      f'{100.*(tar_idx+1) / len(start_indices):.0f}%) | '
                      f'time elapsed: {time.perf_counter() - toc:.2f}s')
                print(f'Total time elapsed: {time.perf_counter() - tic:.2f}s\n')
