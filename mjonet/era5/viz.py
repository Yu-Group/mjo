from matplotlib import pyplot as plt
from matplotlib import colors

import numpy as np
import xarray as xr

__all__ = ['plot_2d']


def plot_2d(*darrays,
            time=0,
            plevels=None,
            figsize: tuple=None,
            extents: list=None,
            subplot_kws=None,
            cbar_kwargs=None,
            **kwargs):
    """Plot xarray.DataArrays that were loaded directly from CIMP ERA5 or processed via `mjonet.era5.preprocess`.

    Parameters
    __________
    darrays: List[xarray.DataArray]
        DataArrays to plot. Should include the entire global grid.
    time: Union[int, str, numpy.datetime64[ns]], optional
        Common time point in each of the `darrays` to plot. Default is the first time point of each array.
    plevels: Union[int, dict], optional
        Pressure levels to plot for each of the `darrays`.
    figsize: Tuple[float, float], optional
        Passed to `matplotlib.pyplot.subplots`.
    extents: List[int], optional
        Longitude and latitude bounds in format [lon_lower, lon_upper, lat_lower, lat_upper]
    subplot_kws: dict, optional
        Passed to `matplotlib.pyplot.subplots`. Default is `{'projection': ccrs.PlateCarree(central_longitude=180)}`.
    cbar_kwargs: dict, optional
        Passed to `xarray.DataArray.plot`. Default is `{'shrink': 0.6}`.
    **kwargs, optional
        Additional keyword args passed to `xarray.DataArray.plot`.
    """

    from cartopy import crs as ccrs

    if isinstance(plevels, dict):
        plevels = { k.lower(): v for k, v in plevels.items() }

    R, C = len(darrays) // 2 + len(darrays) % 2, min(len(darrays), 2)

    if figsize is None:
        figsize = (8*C, 5*R)
    if subplot_kws is None:
        subplot_kws = { 'projection': ccrs.PlateCarree(central_longitude=180) }
    if cbar_kwargs is None:
        cbar_kwargs = { 'shrink': 0.6 }
    if isinstance(time, str):
        time =  np.datetime64(time).astype('datetime64[ns]')

    # ensures the plot has standard global coords
    proj = ccrs.PlateCarree()
    kws = {}
    if subplot_kws['projection'].proj4_params['lon_0']  != 0:
        kws = { 'transform': proj }#, 'transform_first': True }
    kws.update(kwargs)

    fig, axs = plt.subplots(nrows=R, ncols=C,
                            sharex=True, sharey=True,
                            subplot_kw=subplot_kws,
                            figsize=figsize)#, squeeze=False)

    for i, darray in enumerate(darrays):

        var_name = darray.name.lower()

        if 'level' in darray.dims and (plevels is None or var_name not in plevels):
            darray = darray.isel(level=0)
        elif 'level' in darray.dims:
            plevel = plevels[var_name]
            assert isinstance(plevel, int), 'plevels dict must have int values'
            darray = darray.sel(level=plevel)

        if 'forecast_initial_time' in darray.dims:
            step = darray['forecast_hour'].size

            if isinstance(time, int):
                td = time
            elif isinstance(time, np.datetime64):
                td = time - darray['forecast_initial_time'][0].data
                td = td.astype('timedelta64[h]').astype(int) - 1

            fit = darray['forecast_initial_time'][td // step].data
            fh = darray['forecast_hour'][td % step].data

            darray = darray.sel(forecast_initial_time=fit, forecast_hour=fh)
            darray = darray.assign_coords(time=fit + fh.astype('timedelta64[h]'))
            darray = darray.reset_coords(['forecast_initial_time', 'forecast_hour'], drop=True)
        else:
            if isinstance(time, int):
                darray = darray.isel(time=time)
            elif isinstance(time, np.datetime64):
                darray = darray.sel(time=time)

        assert len(darray.dims) == 2, f'{var_name} has unexpected dims after slicing: {darray.dims}'

        ax = plt.subplot(R, C, i + 1)

        if extents is not None:
            ax.set_extent(extents, crs=proj)
        else:
            ax.set_global()
        ax.coastlines()
        ax.gridlines(draw_labels=True, crs=proj)
        darray.plot(subplot_kws=subplot_kws, cbar_kwargs=cbar_kwargs, **kws)

    plt.show()
