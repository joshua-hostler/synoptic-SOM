import os
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import math
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs

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
    def winning_node(self):

        self.diffs = np.subtract(self.current_obs, self.map)
        self.dists = np.linalg.norm(self.diffs, axis=2)
        self.current_winning_node = np.unravel_index(np.argmin(self.dists, axis=None), self.dists.shape)


    #update map weights, current_epoch, learning rate, sigma
    def update(self):
        lamb = self.max_epoch * self.observation_count / math.log10(self.sigma)
        pct = 1 - (self.current_iter / (self.max_epoch * self.observation_count))
        sigma = self.sigma * math.exp(-self.current_iter / lamb)
        lr = self.lr * pct
        for obs in range(self.observation_count):
            self.current_obs = self.records[obs, :]
            self.winning_node()
            self.current_iter += 1
            for node in self.nodes:
                idx = np.unravel_index(node, self.shape)
                squared_norm = ((idx[0]-self.current_winning_node[0]) ** 2) + ((idx[1] - self.current_winning_node[1]) ** 2)
                hck = math.exp(0.0 - (squared_norm) / (sigma * sigma))
                self.map[idx] = self.map[idx] + lr*hck*(self.current_obs - self.map[idx])


    #train the SOM
    def fit(self):
        for i in range(self.max_epoch):
            self.update()
            self.current_epoch += 1
            if self.current_epoch % (self.max_epoch / 10) == 0: print(self.current_epoch)


    def update_lr(self, lr):
        self.lr = lr

    def u_matrix(self):
        um = np.empty(shape=self.shape)

        u = 0
        v = 0
        down = np.linalg.norm(self.map[u + 1, v] - self.map[u, v])
        right = np.linalg.norm(self.map[u, v + 1] - self.map[u, v])
        um[u,v] = (down + right) / 2

        for v in range(1, self.cols-1):
            down = np.linalg.norm(self.map[u + 1, v] - self.map[u, v])
            left = np.linalg.norm(self.map[u, v - 1] - self.map[u, v])
            right = np.linalg.norm(self.map[u, v + 1] - self.map[u, v])
            um[u,v] = (down + left + right) / 3

        v = self.cols-1
        down = np.linalg.norm(self.map[u + 1, v] - self.map[u, v])
        left = np.linalg.norm(self.map[u, v - 1] - self.map[u, v])
        um[u,v] = (down + left) / 2

        for u in range(1, self.rows-1):
            up = np.linalg.norm(self.map[u - 1, v] - self.map[u, v])
            down = np.linalg.norm(self.map[u + 1, v] - self.map[u, v])
            left = np.linalg.norm(self.map[u, v - 1] - self.map[u, v])
            um[u,v] = (up + down + left) / 3

        u = self.rows-1
        up = np.linalg.norm(self.map[u - 1, v] - self.map[u, v])
        left = np.linalg.norm(self.map[u, v - 1] - self.map[u, v])
        um[u,v] = (up + left) / 2

        for v in range(1, self.cols-1):
            up = np.linalg.norm(self.map[u - 1, v] - self.map[u, v])
            left = np.linalg.norm(self.map[u, v - 1] - self.map[u, v])
            right = np.linalg.norm(self.map[u, v + 1] - self.map[u, v])
            um[u,v] = (up + left + right) / 3

        v = 0
        up = np.linalg.norm(self.map[u - 1, v] - self.map[u, v])
        right = np.linalg.norm(self.map[u, v + 1] - self.map[u, v])
        um[u,v] = (up + right) / 2

        for u in range(1, self.rows-1):
            up = np.linalg.norm(self.map[u - 1, v] - self.map[u, v])
            down = np.linalg.norm(self.map[u + 1, v] - self.map[u, v])
            right = np.linalg.norm(self.map[u, v + 1] - self.map[u, v])
            um[u,v] = (up + down + right) / 3

        for u in range(1, self.rows-1):
            for v in range(1,self.cols-1):
                up = np.linalg.norm(self.map[u - 1, v] - self.map[u, v])
                down = np.linalg.norm(self.map[u + 1, v] - self.map[u, v])
                left = np.linalg.norm(self.map[u, v - 1] - self.map[u, v])
                right = np.linalg.norm(self.map[u, v + 1] - self.map[u, v])
                um[u, v] = (up + down + left + right) / 4

        fig, ax = plt.subplots(1, 1)
        fig.suptitle("U-Matrix")
        y = np.arange(self.rows)
        x = np.arange(self.cols)
        ax.pcolormesh(x, y, um, shading='nearest', cmap='inferno')
        ax.invert_yaxis()
        plt.show()

    def mk_labels(self):
        self.labels = np.empty((self.observation_count, 2), dtype=int)
        self.second_best_labels = self.labels
        self.node_count = np.zeros((self.rows, self.cols), dtype=int)
        self.errors = np.empty(self.observation_count, dtype=float)
        te = 0
        qe = 0
        for obs in range(self.observation_count):
            self.current_obs = self.records[obs, :]
            self.winning_node()
            self.labels[obs, :] = self.current_winning_node
            idx = self.current_winning_node
            self.node_count[idx] += 1
            flat_dists = np.ravel(self.dists)
            sbn = np.unravel_index(np.argsort(flat_dists)[1], self.shape)
            self.second_best_labels[obs, :] = sbn
            if np.linalg.norm(np.asarray(sbn) - np.asarray(idx)) > math.sqrt(2):
                te += 1.0
            qe = qe + self.dists[idx]

        self.te = te / self.observation_count
        self.qe = qe / self.observation_count

class plsom(SOM):
    def __init__(self, ds, field_name, events, rows=5, cols=7, lr=1, max_epoch=1000):
        super().__init__(ds, field_name, rows, cols, lr, max_epoch)
        #self.pl_shape = pl_shape #shape of the pressure grid
        self.events = events #list of dates for event analysis
        self.event_count = np.zeros(self.shape, dtype=int)
        #self.events.LOCALDATET = self.events.LOCALDATET.apply(lambda y: dt.strptime(y, '%m/%d/%Y %X'))

    # plot the trained som nodes in a N x M grid
    def plot_nodes(self):
        N = self.rows
        M = self.cols
        plt.rcParams.update({'figure.autolayout': True})
        fig, axs = plt.subplots(N, M, figsize=(11, 8.5))
        fig.suptitle(
            'SOM arrangement of 500hPa geopotentials over Alaska for MJJAS',
            fontsize=20)

        for i in range(N):
            for j in range(M):
                axs[i, j].contourf(self.map[i, j, :].reshape(self.geoshape))

        plt.show()

    #count events corresponding with each som grid
    def count_events(self):
        '''
        for event in events:
            where event_date == obs_date: event_count[obs_node] += 1
        '''
        self.node_count = np.zeros((self.rows, self.cols), dtype=int)
        for obs in range(self.observation_count):
            node = tuple(self.labels[obs])
            obs_day = pd.Timestamp(self.ds.time.values[obs]).to_pydatetime()
            if obs_day.month == 6:
                if obs_day.day >=11:
                    if obs_day.year >= 1986 and obs_day.year <= 2012:
                        strikes = np.where(self.events.LOCALDATET.values == self.ds.time.values[obs])[0].size
                        self.event_count[node] = self.event_count[node] + strikes
                        self.node_count[node] += 1
            elif obs_day.month == 7:
                if obs_day.day <=20:
                    if obs_day.year >= 1986 and obs_day.year <= 2012:
                        strikes = np.where(self.events.LOCALDATET.values == self.ds.time.values[obs])[0].size
                        self.event_count[node] = self.event_count[node] + strikes
                        self.node_count[node] += 1

    def plot_events(self): #plot the events in som node arrangement
        ltgfreq = np.divide(self.event_count, self.node_count)
        ltgfreqm = np.ma.masked_invalid(ltgfreq)
        fig, ax = plt.subplots(1, 1)
        fig.suptitle("Lightning Strikes per day")
        y = np.arange(self.rows)
        x = np.arange(self.cols)
        ax.pcolormesh(x, y, ltgfreq, shading='nearest', cmap='inferno')
        ax.invert_yaxis()
        text_kwargs1 = dict(ha='center', va='center', color='k', fontsize=12)
        for i in range(self.rows):
            for j in range(self.cols):
                if not(np.isnan(ltgfreq[i,j])): plt.text(j, i, f'{int(ltgfreq[i, j])}', **text_kwargs1)
                '''
                txt = plt.text(j, i, f'{int(ltgfreq[i,j])}',
                               path_effects=[pe.withStroke(linewidth=4, foreground="black")],
                               **text_kwargs1)
                '''
        plt.show()


    def save(self, name): #save outputs to a directory at path
        path = os.getcwd()
        midx = pd.MultiIndex.from_product([np.arange(self.rows), np.arange(self.cols), self.ds['longitude'].values, self.ds['latitude'].values], names=['row', 'col', 'lon', 'lat'])
        df = pd.DataFrame(data={'z': self.map.flatten()}, index=midx)
        ds = xr.Dataset.from_dataframe(df)
        ds.to_netcdf(os.path.join(path, name))
        #summary file w/ metrics, summary stats, errors
        #the mapping
        #test commit comment
        #plots








    """
    Next Steps:
    1. how many obs in each node
    2. label obs to each node
        a. error
    3. error(QE, TE, Sammon->TI)
    4. U-Matrix - done
    5. errors per epoch
    6. repeat cassanos' metrics
    8. reorganize class structure
    9. Event based analysis
    lightning meta data: lat, lon, datetime
    10. output the labels
    
    train on MJJAS SOM on full record
    count days/node in duff season
    count lightning per node in duff season
    generate lightning strikes/day for each node duff season.
    """
