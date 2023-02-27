import numpy as np
import xarray as xr
import som
import pandas as pd
import datetime
from datetime import datetime
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from cupyx.profiler import benchmark
#from osgeo import gdal


def main():
    #open file
    fn = r'C:\Users\harpe\PycharmProjects\synoptic-SOM\MJJAS_1979_daily.nc'
    fn2 = 'duff_anom_500z.nc'
    event_fn = r'C:\Users\harpe\PycharmProjects\plsom\Historical_Lightning_1986_2012_ImpactSystem_AlaskaAlbersNAD83.txt'
    ds = xr.open_dataset(fn2)
    events_df = pd.read_csv(event_fn)
    events_df.LOCALDATET = events_df.LOCALDATET.apply(lambda x: datetime.strptime(x, '%m/%d/%Y %X'))
    ltg_df = pd.read_csv(r'Z:\PyProj\PHYS_HW1\day_node_east_int.csv', index_col=0, parse_dates=True)

    #print(ds.sel(time=ltg_df.index.values))
    ds = ds.sel(time=ltg_df.index.values)


    #flatten the pressure data array
    #training_data = np.true_divide(ds_june.slp.values.reshape((2160, 629)), 100.00)

    dim = ds['z'].values.shape[1]*ds['z'].values.shape[2]
    count = ds['z'].values.shape[0]
    print('obs count: ', count)
    print('dim: ', dim)
    vals = ds['z'].values.reshape(count, dim)
    vals = np.append(vals, np.log(ltg_df['sum'].values[:, None]+1), axis=1)
    scaler = StandardScaler()
    vals_scaled = scaler.fit_transform(vals)
    vals_train, vals_test = tts(vals_scaled, test_size=0.2, random_state=42)
    print(vals_test[:,-1])

    '''idx = np.random.randint(count, size=12)
    data = vals[idx,:]'''

    s = som.SOM(rows=3, cols=4, dim=dim+1)

    s.fit(obs_cpu=vals_train, lr=1, epoch=5000, k=1, init_fn=None)
    print(s.nodes.iloc[:, -1])
    pred = s.predict(vals_test)
    sqerr = np.square(vals_test[:,-1] - pred)
    mse = sqerr.mean()
    print(mse)
    s.nodes = s.nodes.iloc[:,:-1]
    s.to_csv('anomaly_5k_34_ltg.csv')
    labels = s.mk_labels(vals_test)
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

