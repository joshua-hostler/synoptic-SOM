import os
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import math
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import cupy as cp

def long_to_180(long):
    return ((long+180) % 360) - 180

def long_to_360(long):
    return long % 360

class SOM():
    """
    Self-Organizing Map Algorithm. Not Parallizable. Next iteration will be Batch SOM.
    http://syllabus.cs.manchester.ac.uk/pgt/2017/COMP61021/reference/parallel-batch-SOM.pdf
    """

    def __init__(self, rows, cols, dim):
        """
        :param ds: has n rows (= number of observations)
               which are dim long (number of features per observation)
        :param rows: rows of the map grid
        :param cols: columns of the map grid
        :param lr: monotonically decreasing learning rate
        :param sigma: neighborhood scaling
        :param max_iter: maximum iterations
        """
        self.rows = rows
        self.cols = cols
        self.dim = dim
        midx = pd.MultiIndex.from_product([np.arange(rows), np.arange(cols)])
        self.nodes = pd.DataFrame(index=midx, columns=np.arange(dim))

    # return the index of the closest node for an observation
    def winning_node(self, v, nodes):

        '''muv = cp.mean(v)
        munodes = cp.mean(nodes, axis=1)

        covmatrix = cp.cov(v, nodes, ddof=0)
        signodes = cp.diag(covmatrix)[1:]
        sigv = covmatrix[0,0]
        covar = covmatrix[0,1:]

        numerator = 2*muv*munodes*covar
        denominator = (cp.square(muv)+cp.square(munodes))*(sigv+signodes)

        SSIM = numerator/denominator
        idx = cp.argmax(SSIM, axis=None)'''
        diffs = cp.subtract(v, nodes)
        dists = cp.linalg.norm(diffs, axis=1)
        idx = cp.argmin(dists, axis=None)
        # idx = np.unravel_index(np.argmin(dists, axis=None), dists.shape)

        return idx

    #train the SOM
    def fit(self, obs_cpu, lr, epoch, k, init_fn=None):
        obs_gpu = cp.asarray(obs_cpu, dtype=cp.float16)
        obs_count = obs_cpu.shape[0]

        if init_fn:
            nodes_cpu = pd.read_csv(init_fn, index_col=[0,1])
            nodes_cpu = nodes_cpu.values
        else:
            nodes_cpu = np.random.choice(obs_cpu.flatten(), size=(self.rows * self.cols, self.dim), replace=False)

        nodes_gpu = cp.asarray(nodes_cpu, dtype=cp.float16)
        sigma_0 = k * max(self.rows,self.cols) / 2
        a = self.nodes.index.to_numpy()
        x = map(np.array, a)
        arr_cpu = np.array(list(x))
        arr_gpu = cp.asarray(arr_cpu)
        print('fitting')
        lamb = epoch / math.log10(sigma_0)

        epochs = np.arange(epoch)
        sigma_cpu = sigma_0 * np.exp(-epochs / lamb)
        pct_cpu = 1 - epochs / epoch
        lr_i_cpu = lr * pct_cpu

        sigma_gpu = cp.asarray(sigma_cpu)
        lr_i_gpu = cp.asarray(lr_i_cpu)
        for i in range(epoch):
            for o in range(obs_count):
                obs_gpu_o = obs_gpu[o,:]
                bmu = self.winning_node(obs_gpu_o, nodes_gpu)
                diffs = arr_gpu - arr_gpu[bmu]
                norms = cp.linalg.norm(diffs, axis=1)
                hck = cp.exp(-cp.square(norms / sigma_gpu[i]) / 2)
                nodes_gpu = nodes_gpu + lr_i_gpu[i]*hck[:, None]*(obs_gpu_o - nodes_gpu)
            print(i, end=' ')
            #if i % (epoch / 10) == 0: print(1-pct)
        self.nodes[:] = nodes_gpu.get()

    def mk_labels(self, obs_cpu):
        obs_count = obs_cpu.shape[0]
        obs_gpu = cp.asarray(obs_cpu)
        nodes = self.nodes.values.astype(float)
        nodes_gpu = cp.asarray(nodes)
        a = self.nodes.index.to_numpy()
        x = map(np.array, a)
        idxs_cpu = np.array(list(x))
        idxs_gpu = cp.asarray(idxs_cpu)
        labels_cpu = np.empty((obs_count, 2), dtype=int)
        labels_gpu = cp.asarray(labels_cpu)

        for o in range(obs_count):
            obs_o = obs_gpu[o, :]
            bmu = self.winning_node(obs_o, nodes_gpu)
            labels_gpu[o, :] = idxs_gpu[bmu]

        return labels_gpu.get()

    def predict(self, obs_cpu, score_func=None):
        '''predicts the last column based on the first n-1 columns.
        if score_func provided also returns score'''

        obs_count = obs_cpu.shape[0]
        obs_gpu = cp.asarray(obs_cpu)
        nodes = self.nodes.values.astype(float)
        nodes_gpu = cp.asarray(nodes)
        a = self.nodes.index.to_numpy()
        x = map(np.array, a)
        idxs_cpu = np.array(list(x))
        idxs_gpu = cp.asarray(idxs_cpu)
        preds_cpu = np.empty((obs_count, 1), dtype=float)
        preds_gpu = cp.asarray(preds_cpu)
        # self.second_best_labels = self.labels
        # self.node_count = np.zeros((self.rows, self.cols), dtype=int)
        # self.errors = np.empty(self.observation_count, dtype=float)
        # te = 0
        # qe = 0
        for o in range(obs_count):
            obs_o = obs_gpu[o, :-1]
            bmu = self.winning_node(obs_o, nodes_gpu[:,:-1])
            preds_gpu[o, :] = nodes_gpu[bmu,-1]

        if score_func:
            score = score_func(preds_gpu, actuals)
            return preds_gpu.get(), score

        return preds_gpu.get()

    def to_csv(self, path): #save SOM to csv at path
        self.nodes.to_csv(path)

    @classmethod
    def from_csv(cls, path):
        '''
        Constructs a SOM object with information from csv at path.
        :param path: Path to csv file.
        :return: Instance of SOM.
        '''
        nodes = pd.read_csv(path, index_col=[0,1])
        rows = nodes.index.levshape[0]
        cols = nodes.index.levshape[1]
        dim = nodes.shape[1]
        obj = cls(rows=rows, cols=cols, dim=dim)
        obj.nodes[:] = nodes.values.astype(float)

        return obj

    def u_matrix(self):
        um = np.empty(shape=(self.rows, self.cols))

        u = 0
        v = 0
        down = np.linalg.norm(self.nodes.loc[u + 1, v] - self.nodes.loc[u, v])
        right = np.linalg.norm(self.nodes.loc[u, v + 1] - self.nodes.loc[u, v])
        um[u,v] = (down + right) / 2

        for v in range(1, self.cols-1):
            down = np.linalg.norm(self.nodes.loc[u + 1, v] - self.nodes.loc[u, v])
            left = np.linalg.norm(self.nodes.loc[u, v - 1] - self.nodes.loc[u, v])
            right = np.linalg.norm(self.nodes.loc[u, v + 1] - self.nodes.loc[u, v])
            um[u,v] = (down + left + right) / 3

        v = self.cols-1
        down = np.linalg.norm(self.nodes.loc[u + 1, v] - self.nodes.loc[u, v])
        left = np.linalg.norm(self.nodes.loc[u, v - 1] - self.nodes.loc[u, v])
        um[u,v] = (down + left) / 2

        for u in range(1, self.rows-1):
            up = np.linalg.norm(self.nodes.loc[u - 1, v] - self.nodes.loc[u, v])
            down = np.linalg.norm(self.nodes.loc[u + 1, v] - self.nodes.loc[u, v])
            left = np.linalg.norm(self.nodes.loc[u, v - 1] - self.nodes.loc[u, v])
            um[u,v] = (up + down + left) / 3

        u = self.rows-1
        up = np.linalg.norm(self.nodes.loc[u - 1, v] - self.nodes.loc[u, v])
        left = np.linalg.norm(self.nodes.loc[u, v - 1] - self.nodes.loc[u, v])
        um[u,v] = (up + left) / 2

        for v in range(1, self.cols-1):
            up = np.linalg.norm(self.nodes.loc[u - 1, v] - self.nodes.loc[u, v])
            left = np.linalg.norm(self.nodes.loc[u, v - 1] - self.nodes.loc[u, v])
            right = np.linalg.norm(self.nodes.loc[u, v + 1] - self.nodes.loc[u, v])
            um[u,v] = (up + left + right) / 3

        v = 0
        up = np.linalg.norm(self.nodes.loc[u - 1, v] - self.nodes.loc[u, v])
        right = np.linalg.norm(self.nodes.loc[u, v + 1] - self.nodes.loc[u, v])
        um[u,v] = (up + right) / 2

        for u in range(1, self.rows-1):
            up = np.linalg.norm(self.nodes.loc[u - 1, v] - self.nodes.loc[u, v])
            down = np.linalg.norm(self.nodes.loc[u + 1, v] - self.nodes.loc[u, v])
            right = np.linalg.norm(self.nodes.loc[u, v + 1] - self.nodes.loc[u, v])
            um[u,v] = (up + down + right) / 3

        for u in range(1, self.rows-1):
            for v in range(1,self.cols-1):
                up = np.linalg.norm(self.nodes.loc[u - 1, v] - self.nodes.loc[u, v])
                down = np.linalg.norm(self.nodes.loc[u + 1, v] - self.nodes.loc[u, v])
                left = np.linalg.norm(self.nodes.loc[u, v - 1] - self.nodes.loc[u, v])
                right = np.linalg.norm(self.nodes.loc[u, v + 1] - self.nodes.loc[u, v])
                um[u, v] = (up + down + left + right) / 4

        fig, ax = plt.subplots(1, 1)
        fig.suptitle("U-Matrix")
        y = np.arange(self.rows)
        x = np.arange(self.cols)
        ax.pcolormesh(x, y, um, shading='nearest', cmap='inferno')
        ax.invert_yaxis()
        plt.show()


class GeoSOM(SOM):
    def __init__(self, rows, cols, dim, lons, lats, attrs=None):
        super().__init__(rows, cols, dim)
        self.lats = lats
        self.lons = lons
        self.geoshape = tuple((lats.shape[0], lons.shape[0]))
        if attrs:
            self.attrs = attrs
        #self.events.LOCALDATET = self.events.LOCALDATET.apply(lambda y: dt.strptime(y, '%m/%d/%Y %X'))

    def to_netCDF(self, path):
        midx = pd.MultiIndex.from_product([np.arange(self.rows),
                                           np.arange(self.cols),
                                           self.lats.values,
                                           self.lons.values], names=['row', 'column', 'latitude', 'longitude'])
        data = self.nodes.values.flatten()

        df = pd.DataFrame(index=midx, data={'field': data})
        ds = xr.Dataset.from_dataframe(df)
        ds.to_netcdf(path=path)

    @classmethod
    def from_netCDF(cls, path):
        ds = xr.open_dataset(path)
        lons = ds['longitude'].values
        lats = ds['latitude'].values
        dim = lons.shape[0]*lats.shape[0]
        rows = ds.dims['row']
        cols = ds.dims['column']
        obj = cls(rows=rows, cols=cols, dim=dim, lons=lons, lats=lats)
        for r in range(rows):
            for c in range(cols):
                obj.nodes.loc[r,c] = ds.field.values[r,c,:,:].reshape(lats.shape[0]*lons.shape[0])
        return obj

    # plot the trained som nodes in a N x M grid
    def plot_nodes(self, path_out=None, colormap='coolwarm', nclevs=12, clevs=None):
        N = self.rows
        M = self.cols

        plt.rcParams.update({'figure.autolayout': True, 'text.usetex': False, 'axes.titlesize': 12})
        fig, axs = plt.subplots(N, M, figsize=(14, 8.5),
                                subplot_kw={'projection': ccrs.AlbersEqualArea(central_longitude=-154,
                                                                               central_latitude=50,
                                                                               standard_parallels=(55, 65))})

        clmin = np.min(self.nodes.values)
        clmax = np.max(self.nodes.values)

        if colormap == 'coolwarm':
            assert nclevs % 2 == 0, "nclevs must be even for diverging colormaps."
            absmax = np.max([np.abs(clmin), np.abs(clmax)])
            clmax = absmax
            clmin = np.negative(absmax)
            cmap = mcolors.LinearSegmentedColormap.from_list(name='red_white_blue',
                                                             colors=[(0, 0, 1),
                                                                     (1, 1, 1),
                                                                     (1, 0, 0)],
                                                             N=nclevs-1)
        if clevs is None:
            clevs = np.linspace(clmin, clmax, nclevs)

        n = 50
        west = 170
        east = 240
        south = 45
        north = 75

        codes = np.full(n * 4, 2)
        codes[0] = 1
        codes[-1] = 79

        aoi = mpath.Path(
            list(zip(np.linspace(west, east, n), np.full(n, north))) + \
            list(zip(np.full(n, east), np.linspace(north, south, n))) + \
            list(zip(np.linspace(east, west, n), np.full(n, south))) + \
            list(zip(np.full(n, west), np.linspace(south, north, n))), codes

        )

        proj2data = ccrs.PlateCarree()._as_mpl_transform(axs[(0,0)]) - axs[(0,0)].transData
        aoi_in_target = proj2data.transform_path(aoi)

        for idx in self.nodes.index:
            z = self.nodes.loc[idx].values.astype('float')
            z = z.reshape(self.geoshape)
            cs = axs[idx].contourf(self.lons, self.lats, z,
                                   clevs,
                                   transform=ccrs.PlateCarree(),
                                   cmap=cmap)
            axs[idx].contour(self.lons, self.lats, z, clevs,
                              transform=ccrs.PlateCarree(),
                              colors='k', linewidths=1.0)
            axs[idx].set_title(f'{idx}')
            axs[idx].set_boundary(aoi_in_target)
            gl = axs[idx].gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                              rotate_labels=False, x_inline=False, y_inline=False,
                              linestyle='--', color='grey', alpha=0.3)
            gl.xlocator = mticker.FixedLocator(long_to_180(np.linspace(west, east, 8)))
            gl.right_labels = False
            gl.top_labels = False
            gl.left_labels = False
            gl.bottom_labels = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            axs[idx].coastlines()
            axs[idx].set_extent((170, 240, 45-1, 75), crs=ccrs.PlateCarree())

        if path_out:
            plt.savefig(path_out)
        plt.show()
        fig, ax = plt.subplots(figsize=(14, 1))
        #fig.subplots_adjust(bottom=0.5)
        cbar = fig.colorbar(cs, cax=ax, orientation='horizontal', cmap=colormap)
        cbar.set_ticks(ticks=clevs, labels=clevs)
        if path_out:
            plt.savefig(path_out[:-4] + '_cbar' + path_out[-4:])
        plt.show()

    def plot_events(self, df): #plot the events in som node arrangement
        sum_by_node = df.groupby(['node']).sum()
        count_by_node = df.groupby(['node']).count()
        ltgfreq = sum_by_node.div(count_by_node)
        ltgfreq = ltgfreq.reindex(index=self.nodes.index)
        stdv = df.groupby(['node']).std()
        stdv = stdv.reindex(index=self.nodes.index)
        fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
        #fig.suptitle("Lightning Strokes per day for Eastern Interior Alaska")
        y = np.arange(self.rows)
        x = np.arange(self.cols)
        ax.pcolormesh(x, y, ltgfreq.values.reshape(self.rows, self.cols), shading='nearest', cmap='coolwarm')
        ax.invert_yaxis()
        text_kwargs1 = dict(ha='center', va='center', color='k', fontsize=18)
        for idx in ltgfreq.index:
            if not(np.isnan(ltgfreq.loc[idx].values)):
                plt.text(idx[1], idx[0], f'{ltgfreq.loc[idx].values.round(2)}', **text_kwargs1)
            '''
            txt = plt.text(j, i, f'{int(ltgfreq[i,j])}',
                               path_effects=[pe.withStroke(linewidth=4, foreground="black")],
                               **text_kwargs1)
            '''
        plt.show()
        #summary file w/ metrics, summary stats, errors
        #the mapping
        #test commit comment
        #plots

    def chi_sq_table(self, df):
        ltgcount = df.groupby(['node']).sum()
        ltgcount = ltgcount.reindex(index=self.nodes.index)
        count = df.groupby(['node']).count().sum()

        sum_by_node = df.groupby(['node']).sum()
        count_by_node = df.groupby(['node']).count()
        ltgfreq = sum_by_node.div(count_by_node)
        ltgfreq = ltgfreq.reindex(index=self.nodes.index)

        print(count)

        rsums = ltgfreq.sum(level=0).values
        csums = ltgfreq.sum(level=1).values
        tsum = ltgcount.sum().sum()
        print(tsum)
        print(tsum/count)
        print(rsums)
        print(csums)

        ecolfreq = self.rows * tsum / count
        print(ecolfreq)
        erowfreq = self.cols * tsum / count
        print(erowfreq)

        rchisq = np.square(rsums - erowfreq.values) / erowfreq.values
        cchisq = np.square(csums - ecolfreq.values) / ecolfreq.values
        chisq = rchisq.sum() + cchisq.sum()

        degfree = (self.rows-1)*(self.cols-1)

        return chisq, degfree

    def SBU(self, v, nodes):
        '''muv = cp.mean(v)
        munodes = cp.mean(nodes, axis=1)

        covmatrix = cp.cov(v, nodes, ddof=0)
        signodes = cp.diag(covmatrix)[1:]
        sigv = covmatrix[0,0]
        covar = covmatrix[0,1:]

        numerator = 2*muv*munodes*covar
        denominator = (cp.square(muv)+cp.square(munodes))*(sigv+signodes)

        SSIM = numerator/denominator
        idx = cp.argsort(SSIM, axis=None)[-2]'''

        diffs = cp.subtract(v, nodes)
        dists = cp.linalg.norm(diffs, axis=1)
        idx = cp.argsort(dists, axis=None)[1]

        return idx

    def TE(self, obs_cpu):
        obs_count = obs_cpu.shape[0]
        obs_gpu = cp.asarray(obs_cpu)
        nodes = self.nodes.values.astype(float)
        nodes_gpu = cp.asarray(nodes)
        a = self.nodes.index.to_numpy()
        x = map(np.array, a)
        idxs_cpu = np.array(list(x))
        idxs_gpu = cp.asarray(idxs_cpu)
        labels_cpu = np.empty((obs_count, 2), dtype=int)
        labels_gpu = cp.asarray(labels_cpu)
        errcount = 0
        for o in range(obs_count):
            obs_o = obs_gpu[o, :]
            bmu = self.winning_node(obs_o, nodes_gpu)
            sbu = self.SBU(obs_o, nodes_gpu)
            if np.linalg.norm(idxs_gpu[bmu]-idxs_gpu[sbu]) > np.sqrt(2): errcount += 1
            labels_gpu[o, :] = idxs_gpu[bmu]

        return errcount/obs_count

    def annual_freq(self, df):
        dfo = df.groupby([df.index.year, 'node']).count()
        idx = pd.MultiIndex.from_product([df.index.year.unique(), df['node'].unique()])
        #print(dfo.index)
        dfo = dfo.reindex(idx, fill_value=0).sort_index()

        midx = pd.MultiIndex.from_product([df.index.year.unique(),
                                           self.nodes.index.get_level_values(0).unique(),
                                           self.nodes.index.get_level_values(1).unique()])
        dfo = pd.DataFrame(data=dfo.values, index=midx)
        dfo = dfo.div(dfo.sum(level=0), axis='index', level=0)
        dfo = dfo.reorder_levels([1,2,0])

        return dfo

    def plot_freq_series(self, df):

        N = self.rows
        M = self.cols
        plt.rcParams.update({'figure.autolayout': True, 'text.usetex': False})
        fig, axs = plt.subplots(N, M, sharex=True, sharey=True, figsize=(11, 8.5))
        for idx in self.nodes.index:
            axs[idx].scatter(df.loc[idx].index, df.loc[idx], color='black', s=12)

        plt.show()

    def plot_freq_event(self, freq, event):
        dfo = event.groupby([event.index.year]).mean()
        dfo = dfo.sort_index()

        freq = freq.sort_index()

        N = self.rows
        M = self.cols
        plt.rcParams.update({'figure.autolayout': True, 'text.usetex': False})
        fig, axs = plt.subplots(N, M, sharex=True, sharey=True, figsize=(11, 7))
        for idx in self.nodes.index:

            axs[idx].scatter(freq.loc[idx].values, dfo.values, color='black', s=12)

        plt.show()

def SSIM(obs_cpu):
    obs_gpu = cp.asarray(obs_cpu, dtype=cp.float16)
    obs_count = obs_cpu.shape[0]
    SSIM = np.zeros((obs_count,obs_count), dtype=np.float16)
    L = obs_gpu.max() - obs_gpu.min()
    c1 = cp.square(.01*L)
    c2 = cp.square(0.03*L)
    means =  cp.mean(obs_gpu, axis=1)
    covmatrix = cp.cov(obs_gpu, ddof=0)
    for o in range(obs_count):
        if o % (obs_count / 10) < 1: print(o / obs_count)
        #obs_gpu_o = obs_gpu[o, :]
        muv = means[o]
        #munodes = cp.mean(obs_gpu[o:, :], axis=1)

        #covmatrix = cp.cov(obs_gpu_o, obs_gpu[o:,:], ddof=0)
        signodes = cp.diag(covmatrix)[o:].copy()
        sigv = covmatrix[o,o].copy()
        covar = covmatrix[o,o:].copy()

        numerator = (2*muv*means[o:] + c1) * (2*covar + c2)
        denominator = (cp.square(muv) + cp.square(means[o:]) + c1)*(sigv + signodes + c2)

        SSIM_o = numerator / denominator
        SSIM[o,o:] = SSIM_o.get().copy()
        SSIM[o:,o] = SSIM[o,o:].copy()

    return SSIM
    #add channel weights as hyper parameters


