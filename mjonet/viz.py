from matplotlib import pyplot as plt
from matplotlib import colors
from cartopy import crs as ccrs

import numpy as np

def plot_era5(*darrays, extent=None, figsize=(10, 20)):
    if extent is None:
        extent = [-75, 85, -15, 75]

    lat = darrays[0]['lat'][:]
    lon = darrays[0]['lon'][:]

    # extract longitude extent (recall that lon ranges from 0 to 360 in the ERA5 data)
    x_idx = np.intersect1d(np.where(lon >= extent[0] + 180), np.where(lon <= extent[1] + 180))
    x = lon[x_idx]

    # extract latitude extent
    y_idx = np.intersect1d(np.where(lat >= extent[2]), np.where(lat <= extent[3]))
    y = lat[y_idx]

    fig, axs = plt.subplots(ncols=4, squeeze=False, figsize=figsize)

    for i, darray in enumerate(darrays):
        proj = ccrs.PlateCarree(central_longitude=-180.0)
        ax = fig.add_subplot(i // 4, i % 4, i, projection=proj)
        ax.set_extent(extent, crs=proj)
        ax.coastlines()
        ax.gridlines(draw_labels=True, crs=proj)
        filled_c = ax.contourf(x, y, darrays[i][y_idx, :][:, x_idx], transform=ccrs.PlateCarree())
        fig.colorbar(filled_c, orientation='vertical')

    plt.show()
