"""See https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/ or
https://towardsdatascience.com/playing-with-time-series-data-in-python-959e2485bff8"""
import os
import re
import dill
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from Functions import config_functions as cfg
import logging
logger = logging.getLogger(__name__)
# logging.getLogger("ding0").setLevel(logging.WARNING)
# pd.set_option('display.max_columns', None, 'display.max_rows', None)


def get_expansion_cost(inputs):
    file = '{}/Grids/CostRES_grid_[{}]_{}.dill'.format(
        cfg.parent_folder, inputs.edisgo_grid, cfg.scenario[inputs.year])
    logger.info('File chose for grid expansion cost data: {}'.format(file))
    if os.path.isfile(file):
        with open(file, 'rb') as ff:
            yearly_expansion = dill.load(ff)
        logger.info('Yearly grid expansion: \n{}'.format(yearly_expansion.head()))
    else:
        logger.info('!!! {} is not yet available !!!'.format(file))
        quit()
    grid_expansion_extreme_day = yearly_expansion.annuities.sort_values(ascending=False).index[0]

    # # Plotexpansion cost breakdown
    # yearly_expansion.sum(axis=1).sort_values(ascending=False).reset_index().plot(legend=False)
    # plt.title('Expansion cost sorted descending by cost occurred from individual days')
    # plt.ylabel('Expansion cost [kEUR]')
    # plt.show()

    # Add one day with expansion cost
    exp_cost = pd.DataFrame(index=pd.date_range("00:00", "23:30", freq="30min").time,
                            columns=pd.date_range("2011-01-01", "2011-12-31", freq="1d"))
    for row in exp_cost.index:
        exp_cost.loc[row, :] = list(yearly_expansion.sum(axis=1))
    exp_cost = exp_cost.set_index(np.repeat('expansion', exp_cost.shape[0]), append=True
                                  ).swaplevel(i=-2, j=-1, axis=0) / exp_cost.max().max()
    exp_cost.columns = exp_cost.columns.astype('str')

    return grid_expansion_extreme_day, exp_cost


def get_gridloads(inputs, aggregation):
    from Functions import grid_functions as grd

    grid = grd.get_grid(inputs)
    demand_load_alldays = \
        pd.DataFrame(grid.pd_load.mean(axis=1), columns=['demand']) if aggregation is True else grid.pd_load
    demand_normal_alldays = demand_load_alldays / demand_load_alldays.max()  # Normalize

    generation_solar = pd.DataFrame(index=inputs.index)
    generation_wind = pd.DataFrame(index=inputs.index)
    for mv_node in grid.edisgo.network.mv_grid.generators:
        if mv_node.type == 'solar':
            generation_solar.loc[:, mv_node.id] = mv_node.timeseries.p
        if mv_node.type == 'wind':
            generation_wind.loc[:, mv_node.id] = mv_node.timeseries.p

    for node in grid.edisgo.network.pypsa.generators_t.p.columns:
        if re.split('_', node)[0] == 'solar':
            generation_solar.loc[:, node] = grid.edisgo.network.pypsa.generators_t.p.loc[:, node]
        if re.split('_', node)[0] == 'wind':
            generation_wind.loc[:, node] = grid.edisgo.network.pypsa.generators_t.p.loc[:, node]

    if aggregation:
        generation_solar = generation_solar.mean(axis=1)
        generation_wind = generation_wind.mean(axis=1)
    generation_solar = generation_solar / generation_solar.max().max()  # Normalize
    try:
        generation_wind = generation_wind / generation_wind.max().max()  # Normalize
    except AttributeError:  # If all wind timesteps are NaN
        generation_wind.loc[:] = 0

    production_normal_alldays = pd.DataFrame(generation_solar, columns=['solar'])
    production_normal_alldays.loc[:, 'wind'] = generation_wind

    # Turn datetime index into multiindex
    demand_normal_alldays.index = pd.MultiIndex.from_tuples(list(zip(demand_normal_alldays.index.date,
                                                                     demand_normal_alldays.index.time)))
    production_normal_alldays.index = pd.MultiIndex.from_tuples(list(zip(production_normal_alldays.index.date,
                                                                         production_normal_alldays.index.time)))

    # Get one vector for each day including all RES and demand nodes
    demand_vector = demand_normal_alldays.T.stack()
    production_vector = production_normal_alldays.T.stack()
    total_vector = demand_vector.append(production_vector)

    return total_vector, grid.nodes['node'], grid


def plot_elbowdata(folder, grid_clusters=False, time_clusters=True):
    from matplotlib.lines import Line2D

    def plot(data, savefile, lim, loc):
        logger.info('Plotting elbow data for: {}'.format(savefile))
        plt.clf()
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        # Plot lines with different marker sizes
        ax1.plot(data.loc[:, 'k'].tolist(), data.loc[:, 'Silhouette score'].tolist(), label='Silhouette score', lw=2,
                 marker='o', ms=7, color='orange')
        ax2.plot(data.loc[:, 'k'].tolist(), data.loc[:, 'Distortion score'].tolist(), label='Distortion score', lw=2,
                 marker='s', ms=7, color='b')
        ax1.set_xlabel('Number of clusters')
        ax1.set_ylabel('Silhouette score')
        ax2.set_ylabel('Distortion score')
        ax2.grid(None)
        plt.xlim([0, lim])
        labels = ['Silhouette score', 'Distortion score']
        styles = ['-', '-']
        markers = ['o', 's']
        colors = ['orange', 'b']
        lines = [Line2D([0], [0], color=colors[i], linestyle=styles[i], marker=markers[i]) for i in range(len(labels))]
        ax1.legend(lines, labels, loc=loc)
        plt.savefig(savefile)
        plt.close()

    sns.set_style("whitegrid")
    if grid_clusters:
        ding0_files = 'C:/Users/Christoph Heilmann/Box Sync/00. Promotion TUM/07 Paper No2/01 External Data/02 Dingo grids'
        name = '2020-06-24'
        file = '{}/Ding0 metadata/{}/Elbow_Data.csv'.format(ding0_files, name)

        data = pd.read_csv(file).loc[:48, :]
        data.columns = ['k', 'Distortion score', 'Silhouette score', 'calinski_harabaz']
        savefile = '{}/Elbow_Plot_Grids.png'.format(folder)
        plot(data, savefile, lim=51, loc='right')
    elif time_clusters:
        for grid in [163, 404, 1791, 1392, 2948, 1944, 55, 1055]:
            try:
                data = pd.read_csv('{}/cluster_timesteps_{}_s2s2000s50.csv'.format(folder, grid))
                data.columns = ['k', 'Distortion score', 'Silhouette score']
                savefile = '{}/Timesteps_elbow_{}_s2s2000s50.png'.format(folder, grid)
                plot(data, savefile, lim=2000, loc='lower right')
            except FileNotFoundError:
                logger.warning('File to plot eblow for cluster {} not found!'.format(grid))
                pass


def get_pev_behaviour(nodes, inputs, grid, aggregation, new_rides=False):
    import os
    import pev_functions as pev

    # Get PEV data
    ride_prob, ta, td, km_from, km_to = {}, {}, {}, {}, {}
    for load_type in ['agricultural', 'industrial', 'residential', 'retail']:
        input_file = '{}/Inputs/PEV_data_{}.pkl'.format(cfg.parent_folder, load_type)
        with open(input_file, 'rb') as fff:
            ride_prob[load_type], ta[load_type], td[load_type], km_from[load_type], km_to[load_type] = dill.load(fff)
    t_arrival = pd.DataFrame(0, index=ta['residential'].index,
                             columns=['agricultural', 'industrial', 'residential', 'retail'])
    t_departure = pd.DataFrame(0, index=ta['residential'].index,
                               columns=['agricultural', 'industrial', 'residential', 'retail'])
    for nodetype in t_arrival.columns:
        t_arrival.loc[:, nodetype] += ta[nodetype].CumSum.diff().rename(nodetype)
        t_departure.loc[:, nodetype] += td[nodetype].CumSum.diff().rename(nodetype)
    t_arrival[t_arrival < 0] = 0
    t_departure[t_departure < 0] = 0

    # Add t_arrival, t_departure for each node and day to integrate PEV behaviour
    t_arrival.index = pd.MultiIndex.from_product([list(range(7)),
                                                  pd.date_range("00:00", "23:30", freq="30min").time])
    t_departure.index = pd.MultiIndex.from_product([list(range(7)),
                                                    pd.date_range("00:00", "23:30", freq="30min").time])
    t_arrival_nodes, t_departure_nodes = pd.DataFrame(), pd.DataFrame()

    rawdata_file = '{}/Grids/PEV_[{}]_rawdata_p{}_y{}.pkl'.format(
        cfg.parent_folder, inputs.edisgo_grid, inputs.pev_share[inputs.year], inputs.year)
    if not os.path.isfile(rawdata_file):
        logger.warning('PEV ride data not found in file {}'.format(rawdata_file))
        pev.get_pev(new_rides, inputs, grid=grid)
    with open(rawdata_file, 'rb') as f:
        rawdata = dill.load(f)

    ride_prob, ta, td, km_from, km_to = pev.get_pevdata()
    for day, day_number in zip(inputs.index[::cfg.t_periods], range(len(inputs.index[::cfg.t_periods]))):
        if not os.path.isfile(rawdata_file):
            rawdata.append(pev.calc_pev_data(inputs, grid.pkw_number, grid.consumption, grid.nodes['node'], ride_prob, ta,
                                             td, km_to, day, return_rides_at_node=False))
            logger.warning('PEV rides for day {} have been generated'.format(day))
        ride_at_node_demandtype = rawdata[day_number].groupby(('demand_type', 'node')).count().arr_t
        arrivals_expected_date, departures_expected_date = pd.DataFrame(), pd.DataFrame()
        for node in nodes:
            try:
                arrivals_expected_date.loc[:, node] = t_arrival.loc[day.weekday(), :].mul(
                    ride_at_node_demandtype.loc(axis=0)[:, node].sum(level=0)).sum(axis=1).rename(node)
                departures_expected_date.loc[:, node] = t_departure.loc[day.weekday(), :].mul(
                    ride_at_node_demandtype.loc(axis=0)[:, node].sum(level=0)).sum(axis=1).rename(node)
            except KeyError:
                logger.info('No PEV found at node {}'.format(node))
                pass

        t_arrival_nodes.loc[:, day] = arrivals_expected_date.T.stack().fillna(0)
        t_departure_nodes.loc[:, day] = departures_expected_date.T.stack().fillna(0)
        logger.info('Got pev data from day {}'.format(day))

    t_arrival_nodes = t_arrival_nodes.mean(axis=0, level=1) if aggregation else t_arrival_nodes
    t_arrival_nodes = t_arrival_nodes.set_index(np.repeat('arrivals', t_arrival_nodes.shape[0]),
                                                append=True).swaplevel(i=-2, j=-1, axis=0)
    t_departure_nodes = t_departure_nodes.mean(axis=0, level=1) if aggregation else t_departure_nodes
    t_departure_nodes = t_departure_nodes.set_index(np.repeat('departures', t_departure_nodes.shape[0]),
                                                    append=True).swaplevel(i=-2, j=-1, axis=0)

    return t_arrival_nodes, t_departure_nodes


def plot_data(load_df, inputs, aggregation):
    import random

    random_days = 150
    selection = random.sample(list(load_df.columns), random_days)
    load_df.loc[:, selection].plot(legend=False, alpha=0.1)
    plt.title('Random selection of {} days'.format(random_days))
    plt.ylabel('Relative amount of demand load, generation load or expansion cost')
    plt.xlabel('(Category, Time of day)')
    plt.savefig('{}/Auswertung/Cluster_[{}]_data_p{}_y{}_agg{}.png'.format(
        cfg.parent_folder, inputs.edisgo_grid, inputs.pev_share[inputs.year], inputs.year, aggregation),
        dpi=500)
    plt.close()


def plot_clusters(load_df, clustering, selection, inputs, aggregation):
    import math

    # Plot selected days
    load_df.loc[:, selection.index].plot(alpha=0.8)
    plt.title('Days selected to represent clusters')
    plt.ylabel('Relative amount of demand load, generation load or expansion cost')
    plt.xlabel('(Category, Time of day)')
    plt.savefig('{}/Auswertung/Cluster_[{}]_daysselected_p{}_y{}_c{}_agg{}.png'.format(
        cfg.parent_folder, inputs.edisgo_grid, inputs.pev_share[inputs.year], inputs.year, selection.shape[0],
        aggregation),
        dpi=500)
    plt.close()

    # Plot clusters
    rows = max(2, math.floor((selection.shape[0] + 1)**.5))
    columns = math.ceil((selection.shape[0] + 1) / rows)
    fig, ax_array = plt.subplots(rows, columns)
    i = 1
    logger.info('Number of clusters to be plotted: {}'.format(selection.shape[0]))
    for ax_row in ax_array:
        column = 0
        while column < columns:
            logger.info('Cluster {}'.format(i))
            ax = ax_row[column]
            if i <= selection.shape[0]:
                main_day = selection[selection.clusters == i].index
                cluster_days = main_day if list(clustering[clustering == i].index) == [] \
                    else clustering[clustering == i].index
                logger.info(cluster_days)
                ax.plot(load_df.loc[:, cluster_days].reset_index(drop=True), alpha=0.1)
                ax.plot(load_df.loc[:, main_day].reset_index(drop=True))
                ax.title.set_text('Selected day: {}, Cluster: {}'.format(main_day[0], i))
            else:
                pass
            i += 1
            column += 1

    plt.savefig('{}/Auswertung/Cluster_[{}]_clusterselectdays_p{}_y{}_c{}_agg{}.png'.format(
        cfg.parent_folder, inputs.edisgo_grid, inputs.pev_share[inputs.year], inputs.year, selection.shape[0],
        aggregation),
        dpi=500)
    plt.close()

    # Plot yearly overview of clustering
    plotting = pd.DataFrame(index=clustering.index, columns=list(range(selection.shape[0])))
    for i in range(selection.shape[0]):
        plotting.loc[clustering[clustering == i].index, i] = i + 1
    plotting.plot.bar(stacked=True, legend=False)
    plt.title('Cluster distribution across a year')
    plt.ylabel('Cluster')
    plt.xlabel('Day of the year')
    plt.xticks([])
    plt.savefig('{}/Auswertung/Cluster_[{}]_yearlyclusterdist_p{}_y{}_c{}_agg{}.png'.format(
        cfg.parent_folder, inputs.edisgo_grid, inputs.pev_share[inputs.year], inputs.year, selection.shape[0],
        aggregation),
        dpi=500)
    plt.close()


def plot_calender_view(cluster, selected_days, inputs, aggregation):
    def calendar_array(dates, data):
        i, j = zip(*[d.isocalendar()[1:] for d in dates])
        i = np.array(i) - min(i)
        j = np.array(j) - 1
        ni = max(i) + 1

        calendar = np.nan * np.zeros((ni, 7))
        calendar[i, j] = data
        return i, j, calendar

    def calendar_heatmap(ax, dates, data):
        i, j, calendar = calendar_array(dates, data)
        # im = ax.imshow(calendar, interpolation='none', cmap='summer')
        label_days(ax, dates, i, j, calendar)
        label_months(ax, dates, i, j, calendar)
        # ax.figure.colorbar(im)

    def label_days(ax, dates, i, j, calendar):
        ni, nj = calendar.shape
        day_of_month = np.nan * np.zeros((ni, 7))
        day_of_month[i, j] = [d.day for d in dates]

        for (i, j), day in np.ndenumerate(day_of_month):
            if np.isfinite(day):
                ax.text(j, i, int(day), ha='center', va='center')

        ax.set(xticks=np.arange(7),
               xticklabels=['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'])
        ax.xaxis.tick_top()

    def label_months(ax, dates, i, j, calendar):
        month_labels = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                                 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        months = np.array([d.month for d in dates])
        uniq_months = sorted(set(months))
        yticks = [i[months == m].mean() for m in uniq_months]
        labels = [month_labels[m - 1] for m in uniq_months]
        ax.set(yticks=yticks)
        ax.set_yticklabels(labels, rotation=90)

    dates = cluster.index.to_datetime()
    data = np.array(cluster)
    fig, ax = plt.subplots(figsize=(3, 10))
    calendar_heatmap(ax, dates, data)
    plt.savefig('{}/Auswertung/Cluster_[{}]_calenderview_p{}_y{}_c{}_agg{}.png'.format(
        cfg.parent_folder, inputs.edisgo_grid, inputs.pev_share[inputs.year], inputs.year, selected_days.shape[0],
        aggregation),
        dpi=500)
    plt.close()


def cluster_ward(load_df, periods):
    z = linkage(load_df.T.iloc[:, 0:periods], 'ward')

    dendrogram(z, leaf_rotation=90, leaf_font_size=12, show_contracted=True)
    # truncate_mode='lastp', p=12, show_leaf_counts=False,
    plt.show()

    # "Elbow diagram
    last = z[-10:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    plt.plot(idxs, last_rev)
    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]
    plt.plot(idxs[:-2] + 1, acceleration_rev)
    plt.show()

    k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
    logger.info("'Suggested number of clusters:", k)
    k = int(input("Please enter the number of clusters: "))
    logger.info("You selected " + str(k) + " clusters")
    clusters = fcluster(z, k, criterion='maxclust')  # clusters = fcluster(Z, max_d, criterion='distance')
    logger.info('Clustering: ' + str(clusters))
    plot_clusters(load_df, clusters, pd.DataFrame(), k)


def kmean_elbow(ding0_files, name, model, x, k=(2, 20), visualize=True):
    from yellowbrick.cluster import KElbowVisualizer

    df = pd.DataFrame(0, index=list(k), columns=['distortion', 'silhouette'])  # , 'calinski_harabaz'])
    for metric in df.columns:
        elbow_plot = '{}/Ding0 metadata/{}/Elbow {}.png'.format(ding0_files, name, metric)
        if not os.path.isfile(elbow_plot):
            logger.info('Starting KElbowVisualizer plot with {} score'.format(metric))
            # Instantiate the KElbowVisualizer with the number of clusters and the metric
            # https://blog.floydhub.com/introduction-to-k-means-clustering-in-python-with-scikit-learn/
            # https://www.scikit-yb.org/en/latest/quickstart.html
            visualizer = KElbowVisualizer(model, k=k, metric=metric, timings=True)
            # distortion, silhouette, calinski_harabaz
            visualizer.fit(x)  # Fit the data
            df.loc[:, metric] = visualizer.k_scores_
            if visualize:
                visualizer.poof(outpath=elbow_plot, clear_figure=True)  # Visualize

    return df


def silhouette_plotting(silhouette_plot, model, x):
    from yellowbrick.cluster import SilhouetteVisualizer

    logger.info('Starting Silhouette plot')
    visualizer = SilhouetteVisualizer(model)
    visualizer.fit(x)  # Fit the data to the visualizer
    visualizer.poof(outpath=silhouette_plot, clear_figure=True)


def cluster_kmneans(inputs, load_df, periods, k=None, plot=False, aggregation=True):
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min

    logger.info('Setting up kmeans cluster model')
    x = load_df.T.iloc[:, 0:periods]
    model = KMeans(random_state=0)

    if plot is True:
        elbow_plot = '{}/Auswertung/Cluster_[{}]_elbow_p{}_y{}_agg{}.png'.format(
            cfg.parent_folder, inputs.edisgo_grid, inputs.pev_share[inputs.year], inputs.year, aggregation)
        if not os.path.isfile(elbow_plot):
            kmean_elbow(elbow_plot, model, x)
    logger.info('Elbow plot for grid {}, pev {}, year {} and agg={} already exists'.format(
        inputs.edisgo_grid, inputs.pev_share[inputs.year], inputs.year, aggregation))

    if k is None:
        k = int(input("Please enter the number of total clusters (including most extreme day): "))
        logger.info("You selected " + str(k) + " clusters")
    k = k - 1  # Subtract one to keep for day with highest expansion need to be included later

    model = KMeans(n_clusters=k).fit(x)
    closest, _ = pairwise_distances_argmin_min(model.cluster_centers_, x)
    representative_days = load_df.iloc[:, closest]
    clusters = model.labels_ + 1

    clustering = pd.Series(clusters, index=load_df.columns)
    selection = pd.DataFrame(clustering.loc[representative_days.columns], columns=['clusters'])
    weights = clustering.value_counts().sort_index()
    selection.loc[:, 'weight'] = list(weights)
    selection.sort_index(inplace=True)

    return k + 1, clustering, selection


def get_clusters(k, inputs, plot=False, aggregation=True, new_rides=False):
    save_file = '{}/Grids/Cluster_[{}]_totalvector_p{}_y{}_agg{}.pkl'.format(
        cfg.parent_folder, inputs.edisgo_grid, inputs.pev_share[inputs.year], inputs.year, aggregation)
    try:
        with open(save_file, 'rb') as ff:
            total_vector, grid_expansion_extreme_day = dill.load(ff)
    except FileNotFoundError:
        load_vector, nodes, grid = get_gridloads(inputs, aggregation)
        grid_expansion_extreme_day, exp_cost = get_expansion_cost(inputs)
        t_arrival_nodes, t_departure_nodes = get_pev_behaviour(nodes, inputs, grid, aggregation, new_rides)

        total_vector = load_vector.append(exp_cost)
        total_vector = total_vector.append(t_arrival_nodes.div(t_arrival_nodes.max().max()))
        total_vector = total_vector.append(t_departure_nodes.div(t_departure_nodes.max().max()))
        total_vector.fillna(0, inplace=True)

        with open(save_file, 'wb') as ff:
            dill.dump([total_vector, grid_expansion_extreme_day], ff)

    if plot:
        cfg.set_design(palette_size=4)
        plot_data(total_vector, inputs, aggregation)

    # cluster_ward(load_df, periods)  # Hierarchy (Ward) oder Partition (kmeans)?
    no_clusters, cluster, selected_days = \
        cluster_kmneans(inputs, total_vector, len(total_vector.index), k, plot, aggregation=aggregation)
    if grid_expansion_extreme_day not in selected_days.index:
        selected_days.loc[grid_expansion_extreme_day, :] = [no_clusters, 1]  # Add most extreme day to selected

    # Save clusters
    inputs.clusters = selected_days.shape[0]
    file = '{}/Grids/Clusters_[{}]_p{}_y{}_c{}.pkl'.format(
        cfg.parent_folder, inputs.edisgo_grid, inputs.pev_share[inputs.year], inputs.year, selected_days.shape[0])
    with open(file, 'wb') as f:
        dill.dump([selected_days, cluster], f)

    # Print results
    if plot:
        cfg.set_design(palette_size=4)
        plot_clusters(total_vector, cluster, selected_days, inputs, aggregation)
        plot_calender_view(cluster, selected_days, inputs, aggregation)
    logger.info('Finished: {}'.format(cluster.head()))
    logger.info('Selected days: {}'.format(selected_days))

    return inputs


def analyze_attributes(ding0_files):
    import pickle

    """
    https://openego.readthedocs.io/en/dev/api/ego.tools.html#module-ego.tools.mv_cluster
    
    Calculates the attributes wind and solar capacity and farthest node
    for all files in ding0_files. Results are written to ding0_files

    Parameters
    ----------
    ding0_files : :obj:`str`
        Path to ding0 files

    """
    base_path = ding0_files

    not_found = []
    tccs = []  # Total Cumulative Capacity of Solar
    tccw = []  # Total Cumulative Capacity of Wind
    fnlvmv = []  # the Farthest Node in both networks (lv and mv) in km
    MV_id_list = []  # Distrct id list
    peak_load = []  # Peak load of the grid
    sum_length_list = []  # Total number of lines in the grid in km
    sum_nodes_list = []  # Total number of nodes with transformers in the grid

    for district_number in list(range(1, 4000)):
        try:
            pickle_name = 'ding0_grids_[{}].pkl'.format(district_number)
            nd = pickle.load(open(os.path.join('{}/Ding0 files'.format(base_path), pickle_name), 'rb'))
            logger.info('District no.', district_number, 'found!')
        except:
            not_found.append(district_number)
            continue

        MV_id = 0
        MV_id = nd._mv_grid_districts[0].id_db

        mv_cum_solar_MV = 0  # Solar cumulative capacity in MV
        mv_cum_wind_MV = 0  # Solar cumulative capacity in MV
        peak_load.append(nd._mv_grid_districts[0].peak_load)

        # cumulative capacity of solar and wind in MV
        for geno in nd._mv_grid_districts[0].mv_grid.generators():
            if geno.type == 'solar':
                mv_cum_solar_MV += geno.capacity
            if geno.type == 'wind':
                mv_cum_wind_MV += geno.capacity

        lvg = 0
        mv_cum_solar_LV = 0
        mv_cum_wind_LV = 0

        # cumulative capacity of solar and wind in LV
        for lvgs in nd._mv_grid_districts[0].lv_load_areas():
            for lvgs1 in lvgs.lv_grid_districts():
                lvg += len(list(lvgs1.lv_grid.generators()))
                for deno in lvgs1.lv_grid.generators():
                    if deno.type == 'solar':
                        mv_cum_solar_LV += deno.capacity
                    if deno.type == 'wind':
                        mv_cum_wind_LV += deno.capacity

        # Total solar cumulative capacity in lv and mv
        total_cum_solar = mv_cum_solar_MV + mv_cum_solar_LV
        # Total wind cumulative capacity in lv and mv
        total_cum_wind = mv_cum_wind_MV + mv_cum_wind_LV

        # append to lists
        tccs.append(total_cum_solar)
        tccw.append(total_cum_wind)

        # The farthest node length from MV substation
        from ding0.core.network.stations import LVStationDing0

        tot_dist = []
        max_length = 0
        max_length_list = []
        length_from_MV_to_LV_station_list = []
        max_of_max = 0
        sum_length = 0
        sum_nodes = 0

        # Calculate total cable length for grid
        for edge in nd._mv_grid_districts[0].mv_grid._graph.edge:
            for line in nd._mv_grid_districts[0].mv_grid._graph.edge[edge].keys():
                sum_length += nd._mv_grid_districts[0].mv_grid._graph.edge[edge][line]['branch'].length / 1000

        # Calculate the total number of nodes with transformers in the grid
        for node in list(nd._mv_grid_districts[0].mv_grid._graph.node.keys()):
            try:
                _ = list(node.transformers())[0]
                sum_nodes += 1
            except (AttributeError, IndexError):
                pass

        # make CB open (normal operation case)
        nd.control_circuit_breakers(mode='open')
        # setting the root to measure the path from
        root_mv = nd._mv_grid_districts[0].mv_grid.station()
        # 1st from MV substation to LV station node
        # Iteration through nodes
        for node2 in nd._mv_grid_districts[0].mv_grid._graph.nodes():
            # select only LV station nodes
            if isinstance(
                    node2,
                    LVStationDing0) and not node2.lv_load_area.is_aggregated:

                # Distance from MV substation to LV station node
                length_from_MV_to_LV_station = nd._mv_grid_districts[0].mv_grid.graph_path_length(
                    node_source=node2,
                    node_target=root_mv) / 1000
                length_from_MV_to_LV_station_list.append(length_from_MV_to_LV_station)

                # Iteration through lv load areas
                for lvgs in nd._mv_grid_districts[0].lv_load_areas():
                    for lvgs1 in lvgs.lv_grid_districts():
                        if lvgs1.lv_grid._station == node2:
                            root_lv = node2  # setting a new root
                            for node1 in lvgs1.lv_grid._graph.nodes():
                                # Distance from LV station to LV nodes
                                length_from_LV_staion_to_LV_node = (lvgs1.lv_grid.graph_path_length(
                                            node_source=node1,
                                            node_target=root_lv) / 1000)

                                # total distances in both grids MV and LV
                                length_from_LV_node_to_MV_substation = (length_from_MV_to_LV_station +
                                                                        length_from_LV_staion_to_LV_node)

                                # append the total distance to a list
                                tot_dist.append(
                                    length_from_LV_node_to_MV_substation)
                            if any(tot_dist):
                                max_length = max(tot_dist)
                                # append max lengths of all grids to a list
                                max_length_list.append(max_length)

                    if any(max_length_list):
                        # to pick up max of max
                        max_of_max = max(max_length_list)

        fnlvmv.append(max_of_max)  # append to a new list
        MV_id_list.append(MV_id)  # append the network id to a new list
        sum_length_list.append(sum_length)
        sum_nodes_list.append(sum_nodes)

        # export results to dataframes
    d = {'id': MV_id_list,
         'Solar_cumulative_capacity': tccs,
         'Wind_cumulative_capacity': tccw,
         'The_Farthest_node': fnlvmv,
         'Peak_load': peak_load,
         'Lines_length': sum_length_list,
         'Nodes_with_trafo': sum_nodes_list}  # assign lists to columns
    # not founded networks
    are_not_found = {'District_files_that_are_not_found': not_found}

    df = pd.DataFrame(d)  # dataframe for results

    # dataframe for not found files id
    df_are_not_found = pd.DataFrame(are_not_found)

    # Exporting dataframe to CSV files
    df.to_csv('{}/Ding0 metadata'.format(base_path) + '/' + 'attributes.csv', sep=',')
    df_are_not_found.to_csv('{}/Ding0 metadata'.format(base_path) + '/' + 'Not_found_grids.csv', sep=',')


def extract_grid_info(folder, grid_id=None, gridinfo=None):
    """
    Extracting the following parameters from the grid temp files:

    avrg_load_lines: Maximale Belastung einer Leitung mit allen Lasten (Verbrauch, Wind und PV) [kW] pro
        MV-Leitungskapazität [kW], Durchschnitt über alle Leitungen
    avrg_load_lines_by_length: Wie avrg_load_lines aber gewichtet nach Leitungslänge [km]
    avrg_load_trafos: Maximale Belastung eines Trafos mit allen Lasten (Verbrauch, Wind und PV) [kW] pro
        MV/LV-Trafokapazität [kW], Durchschnitt über alle Tafos
    avrg_load_trafos_by_size: Wie avrg_load_trafos aber gewichtet nach Trafokapazität [kW]

    load_sum_lines: Verbrauch Gesamtnetz [kWh] pro durchschnittlicher MV-Leistungskapazität [kW]n
    load_sum_lines_by_length: Wie load_share_lines aber Durchschnitt gewichtet nach Länge der Leistungen
    wind_sum_lines: Wie load_share_lines nur für Wind
    pv_sum_lines: Wie load_share_lines nur für PV

    load_peak_lines: Peak Verbrauch Gesamtnetz [kW] pro durchschnittlicher MV-Leistungskapazität [kW]
    load_peak_lines_by_length: Wie peak_share_lines aber Durchschnitt gewichtet nach Leitungslänge [km]
    wind_peak_lines: Wie load_share_lines nur für Wind
    pv_peak_lines: Wie load_share_lines nur für PV

    load_peak_trafo: Peak Verbrauch Gesamtnetz [kW] pro durchschnittlicher MV/LV-Trafokapazität [kW]
    load_peak_trafo_by_size: Wie load_peak_trafo aber Durchschnitt gewichtet nach Trafokapazität [kW]
    pv_peak_trafo: Wie load_peak_trafo aber mit PV.
    zensus_trafo: Anzahl der Einwohner pro durchschnittlicher MV/LV-Trafokapazität [kW]

    Parameters
    ----------
    grids_done : :list:
        List of all grids that are of interest
    gridinfo : :pd.DataFrame:
        Existing parameters that are available for all grids in grids_done
    folder : :obj:`str`
        Location where extracted parameters are saved

    :return: :pd.DataFrame:
        Index: All grids in grids_done, columns are the parameters described above
    """
    from Functions import grid_functions as grd

    save_file = '{}/Specific_additional_grid_parameters.csv'.format(folder)
    parameters = pd.DataFrame([], columns=['avrg_load_lines', 'avrg_load_lines_by_length', 'avrg_load_trafos',
                                           'avrg_load_trafos_by_size', 'load_sum_lines', 'load_sum_lines_by_length',
                                           'wind_sum_lines', 'pv_sum_lines', 'load_peak_lines',
                                           'load_peak_lines_by_length', 'wind_peak_lines', 'pv_peak_lines',
                                           'load_peak_trafo', 'load_peak_trafo_by_size', 'pv_peak_trafo',
                                           'zensus_trafo', 'notes'])
    if not os.path.isfile(save_file):
        parameters.to_csv(save_file)
        done = []
    else:
        data = pd.read_csv(save_file, index_col=0)
        done = data.index
        if grid_id is None:
            return data

    if grid_id not in done:
        file = '{}/Grids/Grid_[{}]_gridobject_t17520_y2035_RESTrue.pkl'.format(cfg.parent_folder, grid_id)
        if os.path.isfile(file):
            with open(file, 'rb') as f:
                edisgo = dill.load(f)[0].edisgo
        else:
            file = '{}/Grids/Grid_[{}]_gridobject_t17520_y2035_RESTrue_temp.dill'.format(cfg.parent_folder, grid_id)
            with open(file, 'rb') as f:
                edisgo = dill.load(f)

        try:
            if gridinfo is None:
                gridinfo = pd.read_csv('{}/Inputs/Metadata_allGrids.csv'.format(cfg.parent_folder))

            s_nom_lines_by_length = edisgo.network.pypsa.lines.s_nom.mul(
                edisgo.network.pypsa.lines.length.div(
                    edisgo.network.pypsa.lines.length.mean()))
            s_nom_trafo_by_size = edisgo.network.pypsa.transformers.s_nom.mul(
                edisgo.network.pypsa.transformers.s_nom.div(
                    edisgo.network.pypsa.transformers.s_nom.mean()))

            # Average load on lines and trafos
            s_lines = pd.DataFrame(grd.calc_apparent_power(edisgo.network.pypsa.lines_t['p0'].values,
                                                           edisgo.network.pypsa.lines_t['q0'].values),
                                   index=edisgo.network.pypsa.lines_t['p0'].index,
                                   columns=edisgo.network.pypsa.lines_t['p0'].columns)
            parameters.loc[grid_id, 'avrg_load_lines'] = s_lines.max().div(edisgo.network.pypsa.lines.s_nom).mean()
            parameters.loc[grid_id, 'avrg_load_lines_by_length'] = s_lines.max().div(
                s_nom_lines_by_length.mean()).mean()

            s_trafo = pd.DataFrame(grd.calc_apparent_power(edisgo.network.pypsa.transformers_t['p0'].values,
                                                           edisgo.network.pypsa.transformers_t['q0'].values),
                                   index=edisgo.network.pypsa.transformers_t['p0'].index,
                                   columns=edisgo.network.pypsa.transformers_t['p0'].columns)
            parameters.loc[grid_id, 'avrg_load_trafos'] = s_trafo.max().div(
                edisgo.network.pypsa.transformers.s_nom).mean()
            parameters.loc[grid_id, 'avrg_load_trafos_by_size'] = s_trafo.max().div(s_nom_trafo_by_size).mean()

            # Grid sum of loads compared to line capacities
            parameters.loc[grid_id, 'load_sum_lines'] = (gridinfo.loc[int(grid_id), 'consumption'] /
                                                         edisgo.network.pypsa.lines.s_nom.mean())
            parameters.loc[grid_id, 'load_sum_lines_by_length'] = (gridinfo.loc[int(grid_id), 'consumption'] /
                                                                   s_nom_lines_by_length.mean())
            parameters.loc[grid_id, 'wind_sum_lines'] = (gridinfo.loc[int(grid_id), 'wind_capacity'] /
                                                         edisgo.network.pypsa.transformers.s_nom.mean())
            parameters.loc[grid_id, 'pv_sum_lines'] = (gridinfo.loc[int(grid_id), 'solar_capacity'] /
                                                       edisgo.network.pypsa.transformers.s_nom.mean())

            # Grid peaks of loads compared to line capacities
            parameters.loc[grid_id, 'load_peak_lines'] = (gridinfo.loc[int(grid_id), 'peak_loadKw'] /
                                                          edisgo.network.pypsa.lines.s_nom.mean())
            parameters.loc[grid_id, 'load_peak_lines_by_length'] = (gridinfo.loc[int(grid_id), 'peak_loadKw'] /
                                                                    s_nom_lines_by_length.mean())
            try:
                parameters.loc[grid_id, 'wind_peak_lines'] = (edisgo.network.mv_grid.peak_generation_per_technology.wind /
                                                              edisgo.network.pypsa.transformers.s_nom.mean())
            except AttributeError:
                parameters.loc[grid_id, 'wind_peak_lines'] = 0
            try:
                parameters.loc[grid_id, 'pv_peak_lines'] = (edisgo.network.mv_grid.peak_generation_per_technology.solar /
                                                            edisgo.network.pypsa.transformers.s_nom.mean())
            except AttributeError:
                parameters.loc[grid_id, 'pv_peak_lines'] = 0

            # Grid peaks of loads compared to trafo capacities
            parameters.loc[grid_id, 'load_peak_trafo'] = (gridinfo.loc[int(grid_id), 'peak_loadKw'] /
                                                          edisgo.network.pypsa.transformers.s_nom.mean())
            parameters.loc[grid_id, 'load_peak_trafo_by_size'] = (gridinfo.loc[int(grid_id), 'peak_loadKw'] /
                                                                  s_nom_trafo_by_size.mean())
            try:
                parameters.loc[grid_id, 'pv_peak_trafo'] = (edisgo.network.mv_grid.peak_generation_per_technology.solar /
                                                            edisgo.network.pypsa.transformers.s_nom.mean())
            except AttributeError:
                parameters.loc[grid_id, 'pv_peak_trafo'] = 0
            parameters.loc[grid_id, 'zensus_trafo'] = (gridinfo.loc[int(grid_id), 'zensus'] /
                                                       edisgo.network.pypsa.transformers.s_nom.mean())
        except FileNotFoundError as e:
            parameters.loc[grid_id, 'notes'] = 'Grid not found due to {}'.format(e)

        # Save output to file
        logger.info('Additional parameters for grid {} calculated'.format(grid_id))
        parameters.to_csv(save_file, mode='a', header=False)
    else:
        logger.info('Additional parameters for grid {} already exist'.format(grid_id))


def load_grid_metadata(ding0_files):
    file = '{}/Ding0 metadata/attributes.csv'.format(ding0_files)
    cluster_base = pd.read_csv(file).set_index('id', drop=True).drop('Unnamed: 0', axis=1)

    temp_file = '{}/Ding0 metadata/oedb_grid_metadata_ego_dp_mv_griddistrict.csv'.format(ding0_files)
    ego_dp_mv_griddistrict = pd.read_csv(temp_file, index_col=[0])
    ego_dp_mv_griddistrict.index.names = ['id']
    cluster_base = pd.concat([cluster_base,
                              ego_dp_mv_griddistrict.loc[:, ['consumption',
                                                             'consumption_per_area',
                                                             'zensus_sum']]], axis=1)

    temp_file = '{}/Ding0 metadata/oedb_grid_metadata_ego_dp_hvmv_substation.csv'.format(ding0_files)
    ego_dp_hvmv_substation = pd.read_csv(temp_file, index_col=[0], encoding="ISO-8859-1")
    ego_dp_hvmv_substation.index.names = ['id']
    cluster_base = pd.concat([cluster_base,
                              ego_dp_hvmv_substation.loc[:, ['lon', 'lat']]], axis=1)

    # This information is only available until grid 3397:
    # temp_file = '{}/Ding0 metadata/oedb_grid_metadata_ego_grid_ding0_mv_grid.csv'.format(ding0_files)
    # ego_grid_ding0_mv_grid = pd.read_csv(temp_file, index_col=[0])
    # ego_grid_ding0_mv_grid.index.names = ['id']
    # cluster_base = pd.concat([cluster_base,
    #                           ego_grid_ding0_mv_grid.loc[:, ['population']]], axis=1)

    # temp_file = '{}/Ding0 metadata/oedb_grid_metadata_ego_dp_lv_griddistrict.csv'.format(ding0_files)
    # ego_dp_lv_griddistrict = pd.read_csv(temp_file, index_col=[0])

    cluster_base = cluster_base.dropna()

    return cluster_base


def get_cluster_base(ding0_files, exclude):
    cluster_base = load_grid_metadata(ding0_files)

    with open('{}/Inputs/grid_pkw_numbers.pkl'.format(cfg.parent_folder), 'rb') as f:
        pkw_data = dill.load(f)
    pkw_data = pkw_data.groupby('grid_id').sum().loc[:, ['PKW_gridPLZ', 'Max_PLZAreaSqKm']]
    pkw_data.index.name = 'id'
    cluster_base = cluster_base.merge(pkw_data, left_index=True, right_index=True)
    cluster_base.columns = ['lines_lengthKm', 'nodes_count', 'peak_loadKw', 'solar_capacity', 'dist_farthest_node',
                            'wind_capacity', 'consumption', 'consumption_per_area', 'zensus', 'lon', 'lat',
                            'pkw_count', 'areaSqKm']
    if exclude is not None:
        cluster_base.drop(exclude, inplace=True)  # Drop grids that are not working

    return cluster_base


def cluster_mv_grids(ding0_files, metadata_path, name='Test', exclude_grids=None, exclude_parameters=None,
                     parameters_only=None, no_clusters=None, plot=False, specific_parameters=True,
                     use_significant_parameters=False):
    """
    https://openego.readthedocs.io/en/dev/api/ego.tools.html#module-ego.tools.mv_cluster

    Clusters the MV grids based on the attributes, for a given number
    of MV grids

    Parameters
    ----------
    ding0_files : :obj:`str`
        Path to ding0 files
    no_clusters : int
        Desired number of clusters (of MV grids)

    Returns
    -------
    :param ding0_files:
    :param plot:
    :param no_clusters:
    :param parameters_only:
    :param exclude_parameters:
    :param exclude_grids:
    :param name:
    :param metadata_path: """
    import copy
    from sklearn.cluster import KMeans

    data_base = get_cluster_base(ding0_files, exclude_grids)
    if specific_parameters:
        data_special = extract_grid_info('{}/{}'.format(metadata_path, name)).drop('notes', axis=1)

        # Remove dublicates from data_special
        data_special = data_special.reset_index()
        data_special = data_special.groupby('index').max()
        data_special.index.name = 'grid'

        data_all = data_base.merge(data_special, how='inner', left_index=True, right_index=True)
    else:
        data_all = data_base

    # Combined parameters
    # datas = ['nodes_count', 'zensus', 'zensus', 'consumption']  # 'lines_lengthKm', 'consumption',
    # bases = ['lines_lengthKm', 'lines_lengthKm', 'nodes_count', 'lines_lengthKm']  # 'nodes_count', 'dist_farthest_node'
    # for parameter in parameters:
    #    if parameter not in cluster_base.columns:
    #        data, base = re.split('_by_', parameters[1])
    #        cluster_base.loc[:, parameter] = cluster_base.loc[:, data].div(cluster_base.loc[:, base])

    if exclude_parameters is not None:
        data_all.drop(exclude_parameters, inplace=True, axis=1)

    if use_significant_parameters:
        import re

        # Select parameters that have shown as having a significant correlation with the savings in previous runs
        parameters_only += ['areaSqKm_by_consumption', 'areaSqKm_by_zensus',
                            'dist_farthest_node_by_consumption',  # 'dist_farthest_node',
                            'lines_lengthKm_by_consumption',  # 'lines_lengthKm',
                            'nodes_count_by_consumption']  # 'nodes_count',

        # Generate missing parameters
        for parameter in parameters_only:
            if parameter not in data_all.columns:
                data_all.loc[:, parameter] = data_all.loc[:, re.split('_by_', parameter)[0]].div(
                    data_all.loc[:, re.split('_by_', parameter)[1]])

        # Select only necessary parameters
        data_all = data_all.loc[:, parameters_only]

    cluster_data = copy.deepcopy(data_all)
    if parameters_only is not None:
        cluster_data = cluster_data.loc[:, parameters_only]

    cluster_data_pu = cluster_data.div(cluster_data.max())

    id_ = []
    m = []
    for idx, row in cluster_data_pu.iterrows():
        id_.append(idx)
        f = []
        for attribute in row:
            f.append(attribute)
        m.append(f)
    X = np.array(m)

    logger.info(
        'Used Clustering Attributes: \n {}'.format(
            list(cluster_data_pu.columns)))

    # Starting KMeans clustering
    ran_state = 1808
    kmeans = KMeans(random_state=ran_state)

    if plot is True:
        kmean_elbow(ding0_files, name, kmeans, X)

        for cs in range(6, 10):
            silhouette_plot = '{}/Ding0 metadata/{}/Silhouette plot_{}.png'.format(ding0_files, name, cs)
            if not os.path.isfile(silhouette_plot):
                model = KMeans(cs, random_state=42)
                silhouette_plotting(silhouette_plot, model, X)

    if no_clusters is None:
        no_clusters = int(input("Please enter the number of total clusters: "))
        logger.info("You selected " + str(no_clusters) + " clusters")

    kmeans = KMeans(n_clusters=no_clusters, random_state=ran_state)
    # Return a label for each point
    cluster_labels = kmeans.fit_predict(X)

    # Centers of clusters
    centroids = kmeans.fit(X).cluster_centers_

    id_clus_dist = {}

    # Iterate through each point in dataset array X
    for i in range(len(X)):
        clus = cluster_labels[i]  # point's cluster id
        cent = centroids[cluster_labels[i]]  # Cluster's center coordinates

        # Distance from that point to cluster's center (3d coordinates)
        dist = (
                       (X[i][0] - centroids[clus][0]) ** 2
                       + (X[i][1] - centroids[clus][1]) ** 2
                       + (X[i][2] - centroids[clus][2]) ** 2) ** (1 / 2)

        id_clus_dist.setdefault(clus, []).append({id_[i]: dist})

    cluster_df = pd.DataFrame(
        columns=[
            'no_of_points_per_cluster',
            'cluster_percentage',
            'the_selected_network_id',
            'represented_grids'])
    cluster_df.index.name = 'cluster_id'

    for key, value in id_clus_dist.items():
        no_points_clus = sum(1 for v in value if v)
        # percentage of points per cluster
        clus_perc = (no_points_clus / len(X)) * 100

        id_dist = {}
        for value_1 in value:
            id_dist.update(value_1)

        # returns the shortest distance point (selected network)
        short_dist_net_id_dist = min(id_dist.items(), key=lambda x: x[1])

        cluster_df.loc[key] = [
            no_points_clus,
            round(clus_perc, 2),
            short_dist_net_id_dist[0],
            list(id_dist.keys())]

    logger.info(cluster_df)
    return cluster_df, no_clusters, cluster_data, data_all


def plot_grid_clusters(ding0_files, clustering, no_clusters):
    def boxplot(x, y, name):
        data = data_all_pu.loc[:, [x, y]].stack().reset_index()
        data.columns = ['id', 'type', 'value']
        data = data.merge(data_all,
                          left_on='id',
                          right_index=True).loc[:, ['id', 'type', 'value', 'cluster_number']]
        sns.boxplot(x="cluster_number", y="value", hue="type", data=data, fliersize=0)
        sns.stripplot(x="cluster_number", y='value', hue="type", data=data,
                      dodge=True, jitter=True, alpha=0.2, color='black')
        selection = [x in list(cluster.the_selected_network_id) for x in data.id]  # Selected grid per cluster
        # Get the ax object to use later.
        ax = sns.stripplot(x="cluster_number", y='value', hue="type", data=data[selection], dodge=True)
        # Get the handles and labels. For this example it'll be 2 tuples of length 4 each.
        handles, labels = ax.get_legend_handles_labels()
        # When creating the legend, only use the first two elements to effectively remove the last two.
        plt.legend(handles[0:2], labels[0:2])
        file = '{}/Ding0 metadata/{}/Cluster_c{}_{}_strip.png'.format(ding0_files, clustering, no_clusters, name)
        plt.savefig(file, dpi=500)
        plt.clf()
        logger.info('{} finished'.format(file))

    def scatterplot(x, y, name):
        sns.scatterplot(x=x, y=y, hue="cluster_number", data=data_all, palette=palette, alpha=0.2)
        selection = [x in list(cluster.the_selected_network_id) for x in
                     cluster_data.index]  # Selected grid per cluster
        ax = sns.scatterplot(x=x, y=y, hue="cluster_number", data=data_all[selection], palette=palette)
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[0:no_clusters+1], labels[0:no_clusters+1])
        file = '{}/Ding0 metadata/{}/Cluster_c{}_{}_scatter.png'.format(ding0_files, clustering, no_clusters, name)
        plt.savefig(file, dpi=500)
        plt.clf()
        logger.info('{} finished'.format(file))

    # Getting and preparing data
    with open('{}/Ding0 metadata/{}/Cluster_df_c{}.dill'.format(ding0_files, clustering, no_clusters), 'rb') as f:
        cluster, cluster_data, data_all = dill.load(f)
    data_all.loc[:, 'cluster_number'] = 0
    data_all_pu = data_all.div(data_all.max())
    for cluster_no in cluster.index:
        cluster_data.loc[cluster.loc[cluster_no, 'represented_grids'], 'cluster_number'] = cluster_no
        data_all.loc[cluster.loc[cluster_no, 'represented_grids'], 'cluster_number'] = cluster_no
    cluster_data.cluster_number = pd.Categorical(cluster_data.cluster_number)
    data_all.cluster_number = pd.Categorical(data_all.cluster_number)

    palette = sns.color_palette("husl", data_all.cluster_number.nunique())
    cfg.set_design(palette_size=no_clusters)

    plots = [("consumption", "areaSqKm", "Consumption"),
             ("lon", "lat", "Location"),
             # ("zensus", "areaSqKm", "Area"),
             ('lines_lengthKm', 'nodes_count', 'Sizing'),
             ('lines_lengthKm', 'dist_farthest_node', 'Sizing_lines'),
             ('nodes_count', 'peak_loadKw', 'Sizing_Nodes'),
             ('wind_capacity', 'solar_capacity', 'RES'),
             # ('zensus', 'pkw_count', 'Population'),
             ('consumption', 'pkw_count', 'Population'),
             # ('avrg_load_lines', 'avrg_load_lines_by_length', 'AvrgLines'),
             # ('avrg_load_trafos', 'avrg_load_trafos_by_size', 'AvrgTrafos'),
             # ('load_peak_lines', 'load_peak_lines_by_length', 'PeakLines'),
             # ('load_peak_trafo', 'load_peak_trafo_by_size', 'PeakTrafos')
             ]

    for var1, var2, plotname in plots:
        try:
            boxplot(var1, var2, plotname)
        except (KeyError, ValueError):
            pass
        try:
            scatterplot(var1, var2, plotname)
        except (KeyError, ValueError):
            pass


def run_cluster_grids(clusters=None):
    ding0_f = 'C:/Users/Christoph Heilmann/Box Sync/00. Promotion TUM/07 Paper No2/01 External Data/02 Dingo grids'
    clustering_name = '2020-06-24'  # '{}'.format(str(datetime.datetime.now().date()))
    if clusters is not None:
        clustering_name = '{}/C{}'.format(clustering_name, clusters)
    metadata_path = '{}/Ding0 metadata'.format(ding0_f)
    exclude = [1246, 2330, 2277]
    # 'avrg_load_lines_by_length', 'avrg_load_trafos_by_size',  # 'avrg_load_lines', 'avrg_load_trafos',
    # 'load_peak_lines_by_length', 'load_peak_trafo_by_size']  # 'load_peak_lines', 'load_peak_trafo'
    parameters = [
        'lines_lengthKm', 'dist_farthest_node', 'nodes_count',
        'peak_loadKw', 'solar_capacity', 'wind_capacity', 'consumption', 'consumption_per_area',
        'pkw_count', ]  # zensus, consumption_per_area
    if clustering_name == '2020-07-07':
        exclude.append(2388)
    elif clustering_name == '2020-07-08':
        parameters = [
            'lines_lengthKm', 'dist_farthest_node', 'nodes_count',
            'peak_loadKw', 'solar_capacity', 'wind_capacity', 'consumption',
            'areaSqKm', 'pkw_count', ]  # zensus, consumption_per_area

    cluster, no_clusters, cluster_data, data_all = cluster_mv_grids(ding0_f, metadata_path,
                                                                    name=clustering_name,
                                                                    exclude_grids=exclude,
                                                                    parameters_only=parameters,
                                                                    exclude_parameters=['lon', 'lat'],
                                                                    specific_parameters=False,
                                                                    use_significant_parameters=False,
                                                                    plot=True,
                                                                    no_clusters=clusters)

    cluster.to_csv('{}/{}/Cluster_df_c{}.csv'.format(metadata_path, clustering_name, no_clusters))
    with open('{}/{}/Cluster_df_c{}.dill'.format(metadata_path, clustering_name, no_clusters), 'wb') as f:
        dill.dump([cluster, cluster_data, data_all], f)
    with open('{}/{}/ClusterBase_df_c{}.txt'.format(metadata_path, clustering_name, no_clusters), 'w') as f:
        print(cluster_data.loc[cluster.the_selected_network_id, :], file=f)

    plot_grid_clusters(ding0_f, clustering_name, no_clusters)


def cluster_timesteps_elbow_plot(loads, plot_file='Elbowtest_plot.png', inputs=None, visualize=True, start=2,
                                 stop=1000, steps=100):
    from sklearn.cluster import KMeans

    model = KMeans(random_state=0)
    data = kmean_elbow(plot_file,
                       name='timesteps',
                       model=model,
                       x=loads,
                       k=range(start, stop, steps),
                       visualize=visualize)
    if inputs is not None:
        data.to_csv('{}_s{}s{}s{}.csv'.format(inputs.out_file, start, stop, steps))


def cluster_timesteps_timesteps_weights(grid, size=100, inputs=None, elbow_plot=False, visualize=True, start=2,
                                        stop=1000, steps=100):
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min

    true, reactive = grid.edisgo.network.pypsa.loads_t.p_set, grid.edisgo.network.pypsa.loads_t.q_set
    apparent_load = (true ** 2 + reactive ** 2) ** 0.5  # kVA
    true, reactive = grid.edisgo.network.pypsa.generators_t.p_set, grid.edisgo.network.pypsa.generators_t.q_set
    apparent_genr = (true ** 2 + reactive ** 2) ** 0.5  # kVA
    loads = pd.concat([apparent_load, apparent_genr], axis=1)
    # Normalize load curves
    loads = loads.div(loads.max(0)).fillna(0)

    if elbow_plot:
        cluster_timesteps_elbow_plot(loads,
                                     plot_file='{}_elbow_plot.png'.format(inputs.out_file),
                                     inputs=inputs,
                                     visualize=visualize,
                                     start=start,
                                     stop=stop,
                                     steps=steps)

    model = KMeans(n_clusters=size).fit(loads)
    closest, _ = pairwise_distances_argmin_min(model.cluster_centers_, loads)

    representative_times = loads.index[closest]
    clusters = model.labels_ + 1

    clustering = pd.Series(clusters, index=loads.index)
    selection = pd.DataFrame(clustering.loc[representative_times],
                             columns=['clusters'])
    weights = clustering.value_counts().sort_index()
    selection.loc[:, 'weight'] = list(weights)
    selection.sort_index(inplace=True)

    return selection


def calculate_pvalues(df, roundby=4, return_coefficients=False):
    from scipy.stats import pearsonr

    df = df.dropna()._get_numeric_data()  # ._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    coefficients = dfcols.transpose().join(dfcols, how='outer')

    for x in df.columns:
        for y in df.columns:
            c, p = pearsonr(df[x], df[y])
            pvalues[x][y] = round(p, roundby)
            coefficients[x][y] = round(c, roundby)

    if return_coefficients:
        return pvalues, coefficients
    else:
        return pvalues


# To run:
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_cluster_grids()
    # for c in range(6, 11):
    #    logger.info('Running grid cluster analysis for {} clusters'.format(c))
    #    run_cluster_grids(clusters=c)
    # logging.basicConfig(level=logging.INFO)
    # inpts = inpt.Inputs()
    # get_clusters(k=None, inputs=inpts, plot=True, aggregation=True)  # Aggregate or not?
