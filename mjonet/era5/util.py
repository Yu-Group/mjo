"""Utility functions for working with ERA5 data.
"""

import os
import glob

import numpy as np
import xarray as xr


__all__ = ['dset_from_wds_sample',
           'get_filepaths',
           'grid_2d',
           'make_glob_dict',
           'make_regridder']


def make_glob_dict(var_names: list=None,
                   exclude_vars: list=None,
                   years=None,
                   era5_dir: str='/global/cfs/projectdirs/m3522/cmip6/ERA5'):
    """Make glob strings for each input variable.

    Parameters
    __________
    var_names: List[str]
        The variable names ("Short Name"), as specified by https://rda.ucar.edu/datasets/ds633.0/.
    exclude_vars: List[str]
        Variables to exclude from the default set.
    years: Iterable[Union[int, str]], optional
        An iterable of years in format 'yyyy' or 'yyyymm'. If None, all years found are included.
    era5_dir: str, optional
        The directory of the CMIP6 ERA5 datasets (no trailing slash). Default is the location on
        Perlmutter.

    Returns
    _______
    glob_dict: dict
        A dictionary with entries like (var_name: glob_list)

    """
    if years is None:
        years = ['*']
    else:
        years = [str(year) + '[0-1][0-9]' if len(str(year)) == 4 else str(year)
                 for year in years]

    sfc = 'e5.oper.an.sfc' # 1 in str.format()
    pl = 'e5.oper.an.pl' # 2 in str.format()
    meanflux = 'e5.oper.fc.sfc.meanflux' # 3 in str.format()

    glob_dict = {
        'sp': os.path.join(era5_dir, '{1}/{0}/{1}.128_134_sp.ll025sc.*.nc'),
        'tcwv': os.path.join(era5_dir, '{1}/{0}/{1}.128_137_tcwv.ll025sc.*.nc'),
        'msl': os.path.join(era5_dir, '{1}/{0}/{1}.128_151_msl.ll025sc.*.nc'),
        '10u': os.path.join(era5_dir, '{1}/{0}/{1}.128_165_10u.ll025sc.*.nc'),
        '10v': os.path.join(era5_dir, '{1}/{0}/{1}.128_166_10v.ll025sc.*.nc'),
        '2t': os.path.join(era5_dir, '{1}/{0}/{1}.128_167_2t.ll025sc.*.nc'),
        'z': os.path.join(era5_dir, '{2}/{0}/{2}.128_129_z.ll025sc.*.nc'),
        't': os.path.join(era5_dir, '{2}/{0}/{2}.128_130_t.ll025sc.*.nc'),
        'u': os.path.join(era5_dir, '{2}/{0}/{2}.128_131_u.ll025uv.*.nc'),
        'v': os.path.join(era5_dir, '{2}/{0}/{2}.128_132_v.ll025uv.*.nc'),
        'r': os.path.join(era5_dir, '{2}/{0}/{2}.128_157_r.ll025sc.*.nc'),
        'mtnlwrf': os.path.join(era5_dir, '{3}/{0}/{3}.235_040_mtnlwrf.ll025sc.*.nc')
    }

    glob_dict = { var_name: [glob_str.format(year, sfc, pl, meanflux) for year in years]
                  for var_name, glob_str in glob_dict.items() }

    if var_names is None:

        if exclude_vars is None:
            # return default glob_dict
            return glob_dict

        var_names = glob_dict.keys()

    else:
        var_names = [var_name.lower() for var_name in var_names]

    if exclude_vars is None:
        exclude_vars = []

    exclude_vars = [var_name.lower() for var_name in exclude_vars]

    return { var_name: glob_dict[var_name] for var_name in var_names
             if var_name not in exclude_vars }


def get_filepaths(glob_dict: dict=None):
    """Get file paths for each input variable. Default settings make many assumptions about the
    directory/filepath structure.

    Parameters
    __________
    glob_dict: Dict[str, str], optional
        A dictionary with entries like (var_name: glob_str) where values are glob strings to be
        passed to glob.glob() to find the necessary files, or an iterable of such strings.

    Returns
    _______
    filepath_dict: dict
        A dictionary with entries like (var_name: filepath_list)

    """
    if glob_dict is None:
        glob_dict = make_glob_dict()
    filepath_dict = {}
    var_list = list(glob_dict.keys())
    for var in var_list:
        globber = glob_dict[var]
        filepaths = []
        for glob_str in globber:
            filepaths.extend(glob.glob(glob_str))
        filepaths.sort()
        filepath_dict[var] = filepaths
    return filepath_dict


def grid_2d(obj,
            resolution):
    """Like xesmf.util.grid_2d, but creates a dataset that matches the CMIP ERA5 grid. See
    https://github.com/JiaweiZhuang/xESMF/blob/master/xesmf/util.py.

    obj
        A dataset or data array.
    resolution: float
        The vertical / horizontal spacing between grid points, in degrees.
    """
    # bounds
    # latitude ranges from high to low in ERA5
    lat_b = np.arange(obj['lat'].max(), obj['lat'].min() - resolution, -resolution)
    lon_b = np.arange(obj['lon'].min(), obj['lon'].max() + resolution, resolution)

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


def make_regridder(obj,
                   resolution=1.0,
                   weights_dir=None,
                   clean_weights=False):
    """Return an `xesmf.Regridder` that can regrid input object to the specified resolution.
    Parameters
    __________
    obj
        A dataset or data array. Must have `variables` attribute with `lat` and `lon`, or
        `latitude` and `longitude` if `rename_geo_coords` is True.
    resolution: float, optional
        The vertical / horizontal spacing between grid points, in degrees.
    weights_dir: str, optional
        A directory to look in for existing regridder weights or in which to save the weights if
        none found.
    clean_weights
        If True, clean out the regridder weights file.

    """
    import xesmf as xe

    # regrid
    grid = grid_2d(obj, resolution)

    # reuse weights if possible
    if weights_dir is not None:
        # use existing weights or create and save new ones
        # TODO: better filename
        fname = f'era5_0.25res_regrid_to_{resolution}res_conservative.nc'
        fpath = os.path.join(weights_dir, fname)
        weights_exist = os.path.exists(fpath)
        regridder = xe.Regridder(obj,
                                 grid,
                                 'conservative',
                                 reuse_weights=weights_exist,
                                 filename=fpath)
        if weights_exist and clean_weights:
            regridder.clean_weight_file()
    else:
        # create a new regridder without saving weights
        regridder = xe.Regridder(obj, grid, 'conservative')

    return regridder


def dset_from_wds_sample(sample):
    """Create an `xarray.DataArray` from a WebDataset sample.
    """
    coords = { 'time': sample['time'].astype('datetime64[h]'),
               'lon': sample['lon'],
               'lat': sample['lat'],
               'lon': sample['lon'],
               'variable': sample['variable'] }
    darray = xr.DataArray(sample['data'], coords, sample['dims'])
    return darray.to_dataset(dim='variable')
