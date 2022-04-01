from matplotlib import pyplot as plt
from matplotlib import colors
from cartopy import crs as ccrs

import numpy as np


__all__ = ['plot']


def plot(*darrays, extent=None):
    if extent is None:
        extent = [-75, 85, -15, 75]

    lat = darrays[0]['lat'][:]
    lon = darrays[0]['lon'][:]
    ratio = lon.size / lat.size

    # extract longitude extent (recall that lon ranges from 0 to 360 in the ERA5 data)
    x_idx = np.intersect1d(np.where(lon >= extent[0] + 180), np.where(lon <= extent[1] + 180))
    x = lon[x_idx]

    # extract latitude extent
    y_idx = np.intersect1d(np.where(lat >= extent[2]), np.where(lat <= extent[3]))
    y = lat[y_idx]

    R, C = len(darrays) // 2 + len(darrays) % 2, min(len(darrays), 2)
    figsize = (8*C, 5*R)
    proj = ccrs.PlateCarree(central_longitude=-180.0)
    fig, axs = plt.subplots(nrows=R, ncols=C, 
                            sharex=True, sharey=True, 
                            subplot_kw={'projection': proj}, 
                            figsize=figsize)#,squeeze=False)

    for i, darray in enumerate(darrays):
        # import pdb; pdb.set_trace()
        ax = plt.subplot(R, C, i + 1)
        ax.set_extent(extent, crs=proj)
        ax.coastlines()
        ax.gridlines(draw_labels=True, crs=proj)
        ax.title.set_text(darray.short_name)
        filled_c = ax.contourf(x, y, darray[y_idx, :][:, x_idx], transform=ccrs.PlateCarree())
        plt.colorbar(filled_c, orientation='vertical', ax=ax, fraction=ratio*0.046, pad=0.04)

    plt.show()
