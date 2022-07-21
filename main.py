import numpy as np
import xarray as xr
import som
import pandas as pd
import datetime
from datetime import datetime
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cupyx.profiler import benchmark
#from osgeo import gdal


def main():
    #open file
    fn = r'C:\Users\harpe\PycharmProjects\synoptic-SOM\MJJAS_1979_daily.nc'
    event_fn = r'C:\Users\harpe\PycharmProjects\plsom\Historical_Lightning_1986_2012_ImpactSystem_AlaskaAlbersNAD83.txt'
    ds = xr.open_dataset(fn)
    events_df = pd.read_csv(event_fn)
    events_df.LOCALDATET = events_df.LOCALDATET.apply(lambda x: datetime.strptime(x, '%m/%d/%Y %X'))


    #flatten the pressure data array
    #training_data = np.true_divide(ds_june.slp.values.reshape((2160, 629)), 100.00)

    dim = ds['z'].values.shape[1]*ds['z'].values.shape[2]
    count = ds['z'].values.shape[0]
    print('obs count: ', count)
    print('dim: ', dim)
    vals = ds['z'].values.reshape(count, dim)

    idx = np.random.randint(count, size=35)
    data = vals[idx,:]

    mymap = som.SOM(rows=5, cols=7, dim=dim)

    print(benchmark(mymap.fit(obs_cpu=vals, lr=1, epoch=1000), n_repeat=1))
    #mymap.to_csv('20_000epoch_gpu.csv')
    labels = mymap.mk_labels(vals)
    print(labels)

    secondmap = som.SOM.from_csv('test.csv')
    labels = secondmap.mk_labels(vals)
    print(labels)

    print(mymap.nodes == secondmap.nodes)

    mymap.plot_nodes()

    print(mymap.node_count)
    mymap.count_events()
    #mymap.plot_events()

    print(np.sum(mymap.event_count), np.sum(mymap.node_count))
    print(np.around(np.divide(mymap.event_count,mymap.node_count), decimals=0))
    print(mymap.event_count)

    print(mymap.node_count)
    print(mymap.te, mymap.qe)
    mymap.save('test.nc')

    #plot with coastlines:
    N = mymap.rows
    M = mymap.cols
    plt.rcParams.update({'figure.autolayout': True})
    fig, axs = plt.subplots(N, M, figsize=(11, 8.5),
                            subplot_kw={'projection': ccrs.Mercator(central_longitude=190.0, min_latitude=40.0, max_latitude=75.0)})
    fig.suptitle('SOM arrangement of 500hPa Geopotential Heights over Alaska for MJJAS',
                 fontsize=20)

    clevs=np.linspace(np.min(mymap.map[:,:,:]), np.max(mymap.map[:,:,:]), 12)

    for i in range(N):
        for j in range(M):
            #data,lons=add_cyclic_point(mymap.map[i,j,:].reshape(17,37), coord=ds_june['lon'])
            cs=axs[i, j].contourf(ds['longitude'], ds['latitude'], mymap.map[i,j,:].reshape(121,281), clevs,
                                  cmap='inferno',
                                  transform = ccrs.PlateCarree())
            axs[i, j].set_title(f'{i}, {j}')
            #axs[i, j].set_yticks(ds_june['lat'], crs=ccrs.PlateCarree())
            #axs[i, j].set_xticks(ds_june['lon'], crs=ccrs.PlateCarree())
            axs[i, j].coastlines()
            axs[i, j].set_extent((170, 240, 45, 75))

    #plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9, wspace=0.02, hspace=0.02)
    #cbar_ax = fig.add_axes([0.2, 0.2, 0.6, 0.02])
    #cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')

    plt.show()

    fig, ax = plt.subplots(figsize=(11,1))
    fig.subplots_adjust(bottom=0.5)
    cbar = fig.colorbar(cs, cax=ax, orientation='horizontal', label='z (meters)')
    plt.show()

    mymap.plot_events()
if __name__ == '__main__':
    main()

