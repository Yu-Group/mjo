"""Methods to create and load Webdatasets.
"""

import io
import time

import numpy as np
import webdataset as wds

from .preproc import preprocess

__all__ = ['create_wds',
           'decode_npz']


def _encode_era5(dset):
    """Save a small xr.Dataset to bytes in .npz format with entries for the ndarray and metadata
    dictionary, and return it as a string.
    """
    with io.BytesIO() as stream:
        data = {
            'ndarray': dset.to_array().to_numpy(),
            'time': dset['time'].data,
            'meta': dset.to_dict(data=False)
        }
        np.savez(stream, **data)

        return stream.getvalue()


def _decode_npz(data):
    return np.load(io.BytesIO(data), allow_pickle=True)


decode_npz = wds.handle_extension("npz", _decode_npz)


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
    """Create a WebDataset by saving a specified number of data samples in a single tar file named
    like f'era5-{stage}-000000.tar'.
    
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

    # convert target_steps to sorted indices along dset 'time' axis
    target_steps = np.array(target_steps)
    target_steps.sort()
    target_steps = target_steps // time_step

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
        indices = range(0, len(times) - target_steps[-1], samples_per_tar)

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
            for i in range(subset['time'].size - target_steps[-1]):

                x = subset.isel({'time': [i]})
                y = subset.isel({'time': i + target_steps})

                sink.write({
                    '__key__': f'sample{start_idx + i:06d}',
                    'input.npz': _encode_era5(x),
                    'output.npz': _encode_era5(y)
                })

                x.close()
                y.close()

            subset.close()
            sink.close()

            if verbose:
                print(f'\rCreated tar file {tar_fname} ({tar_idx+1}/{len(indices)} : '
                      f'{100.*(tar_idx+1) / len(indices):.0f}%) | '
                      f'time taken: {time.perf_counter() - tic:.2f} s')