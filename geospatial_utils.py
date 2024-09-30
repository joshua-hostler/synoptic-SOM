import pandas as pd
import xarray as xr
from datetime import datetime
import numpy as np
import xarray as xr
import pandas as pd
import datetime
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

AK_ALBERS_SUBPLOT_KW = {'projection': ccrs.AlbersEqualArea(central_longitude=-154,
                                                           central_latitude=50,
                                                           standard_parallels=(55, 65))}


def subseason(start, end):
    '''
    This function returns an interannual subseason date range as a pandas DatetimeIndex generated
    from start and end Datetime objects. Useful for indexing a subseason from a larger dataset.
    :param start: Datetime object
    :param end: Datetime object
    :return: pandas DatetimeIndex

    Future plans:
    - support for date-like strings similar to pandas daterange pattern.
    '''
    # Create list of interannual starting dates.
    starts = pd.date_range(start, periods=end.year- start.year + 1, freq=pd.DateOffset(years=1))

    # initializes the pandas DatetimeIndex.
    idx = pd.DatetimeIndex([])

    # for each day in starts, generate the subseasonal date range.
    for s in starts:
        # temporary end date for current range.
        temp = datetime(s.year, end.month, end.day)

        # range of dates from s to temp
        r = pd.date_range(s, temp)

        # combine r into idx
        idx = idx.union(r)
    return idx

def anomaly(ds, gb ='time.dayofyear', kind='temporal'):
    '''

    :param ds: xarray DataSet Object
    :param gb: groupby term
    :param kind:
    'temporal':
        removes mean over group's time dimension
    :return: xarray object of specified anomaly kind
    '''

    if kind == 'temporal':
        anom = ds.groupby(gb) - ds.groupby(gb).mean(dim='time')

    elif kind == 'spatiotemporal':
        anom = ds.groupby(gb) - ds.groupby(gb).mean()

    elif kind == 'subseason':
        anom = ds - ds.mean(dim='time')

    elif kind == 'spatial':
        anom = ds - ds.mean(dim=['latitude', 'longitude'])

    elif kind == 'stdscale':
        clim = ds.groupby(gb).mean(dim='time')
        std = ds.groupby(gb).std(dim='time')
        anom = xr.apply_ufunc(lambda x, m, s: (x - m) / s,
                              ds.groupby(gb), clim, std)
    else:
        print(kind + " is not valid kind")
        anom = None

    return anom

def long_to_180(long):
    return ((long+180) % 360) - 180

def long_to_360(long):
    return long % 360

def akplots(ds, f, cmap='coolwarm', nclevs=12, clevs=None):
    fig, ax = plt.subplots(figsize=(11,8.5),
                           subplot_kw=AK_ALBERS_SUBPLOT_KW)

    west = 170
    east = 240
    south = 45
    north = 75


    ax.set_boundary(area_of_interest(ax, west, east, south, north))

    clmin = np.min(ds[f].values)
    clmax = np.max(ds[f].values)

    if cmap == 'coolwarm':
        absmax = np.max([np.abs(clmin), np.abs(clmax)])
        clmax = absmax
        clmin = np.negative(absmax)
        cmap = mcolors.LinearSegmentedColormap.from_list(name='red_white_blue',
                                                         colors=[(0, 0, 1),
                                                                 (1, 1, 1),
                                                                 (1, 0, 0)],
                                                         N=nclevs - 1)

    if clevs is None:
        clevs = np.linspace(clmin, clmax, nclevs)

    cs = ax.contourf(ds.longitude, ds.latitude, ds[f], clevs,
                     transform = ccrs.PlateCarree(),
                     cmap=cmap, alpha=1)
    ax.contour(ds.longitude, ds.latitude, ds[f], clevs,
               transform=ccrs.PlateCarree(), colors='k', linewidths=1.0)

    ax.set_extent((west, east, south-1, north), crs=ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      rotate_labels=False, x_inline=False, y_inline=False,
                      linestyle='--', color='grey')
    gl.xlocator = mticker.FixedLocator(long_to_180(np.linspace(west, east, 8)))
    gl.right_labels = False
    gl.top_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    ax.coastlines()


    plt.show()
    fig, ax = plt.subplots(figsize=(11, 0.5))


    cbar = fig.colorbar(cs, cax=ax, orientation='horizontal', cmap=cmap)
    cbar.set_ticks(ticks=clevs, labels=np.round(clevs))

def area_of_interest(ax, west=170, east=240, south=45, north=75):
    n = 50

    codes = np.full(n*4, 2)
    codes[0] = 1
    codes[-1] = 79


    aoi = mpath.Path(
        list(zip(np.linspace(west,east, n), np.full(n, north))) + \
        list(zip(np.full(n, east), np.linspace(north, south, n))) + \
        list(zip(np.linspace(east, west, n), np.full(n, south))) + \
        list(zip(np.full(n, west), np.linspace(south, north, n))), codes

    )

    proj2data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
    aoi_in_target = proj2data.transform_path(aoi)
    return aoi_in_target
