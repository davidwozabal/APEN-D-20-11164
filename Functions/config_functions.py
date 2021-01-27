import os
import pandas as pd
import logging
logger = logging.getLogger(__name__)

"""
Thetas:
# smart_cc_share
# tc_0 - smart charging as share of grid problems,
# tc_1 - average charging as share of node_free,
# td_0 - smart discharging as share of grid problems,
# tv_0 - remaining time relative to cfg.t_periods,
# tv_1 - average_charge_load_for_remaining_time,
"""

# Grid variables
edisgo_mode = 'mv'
load_types = ['agricultural', 'industrial', 'residential', 'retail']
scenario = {2035: 'nep2035',
            2050: 'ego100'}
employees = pd.Series([0.0000747, 0.0000282, 0, 0.0000280], index=load_types)  # Employees per MWh
linear_pf = False
save_grid_files = True
remove_grid_temp = True
remove_scenario_files = True
remove_outputs_test = True
remove_test_timeseries = True  # Change to True for simulations
waittime = 600  # Wait time for repeated tries of oedb imports
oedb_acces_tries = 100  # Number of tries to access oedb database
res_scenario = {0: 0,
                1: 1,
                2: .05}  # RES scenarios as share of original RES load

# load factors = Sicherheitsfaktoren, see
# https://github.com/openego/eDisGo/blob/dev/edisgo/config/config_grid_expansion_default.cfg#L69-L80
# Source: Rehtanz et. al.: "Verteilnetzstudie für das Land Baden-Württemberg", 2017.
mv_load_case_transformer = 0.5
mv_load_case_line = 0.5
mv_feedin_case_transformer = 1.0
mv_feedin_case_line = 1.0
lv_load_case_transformer = 1.0
lv_load_case_line = 1.0
lv_feedin_case_transformer = 1.0
lv_feedin_case_line = 1.0

# Grid analyze with pypsa
del_error_timesteps = True  # If False find_max_factor is used
factor_total = True  # Should always be true if del_error_timesteps is True
if del_error_timesteps is True:
    assert factor_total is True

# Cost variables
cost_types = ['annuities_grid', 'cost_curtail', 'cost_charge_loss', 'cost_battery_degr',
              'annuities_charge', 'cost_communication', 'cost_grid_losses']
# Low RES cost, Standard, High RES cost, 10% RES cost, low cost of losses, high cost of losses
cost_energy = {'solar':    {2035: [56.0,  70.0,  84.0,  7.0,  70.0,  70.0],
                            2050: [32.0,  40.0,  48.0,  4.0,  40.0,  40.0]},
               'wind':     {2035: [80.0, 100.0, 120.0, 10.0, 100.0, 100.0],
                            2050: [60.0,  75.0,  90.0,  7.5,  75.0,  75.0]},
               'run':      {2035: [68.0,  85.0, 102.0,  8.5,  85.0,  85.0],
                            2050: [92.0, 115.0, 138.0, 11.5, 115.0, 115.0]},  # 'run_of_river_hydro'
               'charging': {2035: [45.0,  45.0,  45.0, 45.0,  45.0,  45.0],
                            2050: [45.0,  45.0,  45.0, 45.0,  45.0,  45.0]},
               'losses':   {2035: [45.0,  45.0,  45.0, 45.0,  36.0,  54.0],
                            2050: [45.0,  45.0,  45.0, 45.0,  36.0,  54.0]}}  # [EUR per MWh]
# Battery cost: https://about.bnef.com/blog/behind-scenes-take-lithium-ion-battery-prices/
# https://www.mobilityhouse.com/media/productattachments/files/The-Mobility-House_Ladestationshersteller-im-Vergleich_2019-01_V4.pdf
# https://www.link-labs.com/blog/cellular-iot

# Investment on top of UCC charger & yearly communication cost [EUR]
# Low, medium and high cost for each category
cost_infra = {'UCC':           {2035: [  0.0,   0.0,   0.0, 0.0],  2050: [  0.0,   0.0,   0.0, 0.0]},  # [EUR]
              'UCC_delayed':   {2035: [  0.0,   0.0,   0.0, 0.0],  2050: [  0.0,   0.0,   0.0, 0.0]},  # [EUR]
              'G2V':           {2035: [ 64.0,  80.0,  96.0, 8.0],  2050: [ 56.0,  70.0,  84.0, 7.0]},  # [EUR]
              'V2G':           {2035: [128.0, 160.0, 192.0, 16.0], 2050: [112.0, 140.0, 168.0, 14.0]},  # [EUR]
              'communication': {2035: [  6.4,   8.0,   9.6, 8.0],  2050: [  4.8,   6.0,   7.2, 6.0]},  # [EUR per year]
              'battery':       {2035: [ 46.4,  58.0,  69.6, 58.0], 2050: [ 32.0,  40.0,  48.0, 40.0]}}  # [EUR per kWh]
# bidirektionalen CHAdeMO Ladestation von Endesa, Enel V2G-Ladestationen:
# https://www.schnellladen.ch/Content/Images/uploaded/produkte/Greenmotion/RangeXT/RangeXT-ProductSheet-DE.pdf
# http://50komma2.de/ww/2018/10/19/bidirektionale-e-ladestation-fuer-zu-hause/
# https://www.electrive.net/2018/10/16/newmotion-stellt-neue-generation-einer-v2g-ladestation-vor/
# https://www.link-labs.com/blog/cellular-iot
invest_years = {'grid':     20,
                'trafo':    20,
                'line':     25,
                'charging': 10}
# http://www.waldlandwelt.de/cgi-bin/afa-tabellen.pl?Energie-%20und%20Wasserversorgung;2
interest = {'grid': 0.07,
            'trafo': 0.07,
            'line': 0.07,
            'charging': 0.03}
# https://www.vreg.be/sites/default/files/Tariefmethodologie/2021-2024/europe_economics_report_v6.pdf
inflation = 0.014432
# http://www.inflationsrate.com/
splits = {2035: 10, 2050: 20}

# Time variables
t_periods = 48  # Time periods in a day
index = pd.date_range('2011-01-01',
                      periods=(365 * t_periods),
                      freq=str(24 / t_periods) + 'H')
dayindex = pd.date_range('2011-01-01',
                         periods=365,
                         freq='d')

# PEV variables
soc_mean = .5
soc_std = .25
scenarios = [{2035: .2, 2050: .4},
             {2035: .3, 2050: .5},
             {2035: .4, 2050: .6},
             {2035: .0, 2050: .0},
             {2035: .05, 2050: .05}]  # PEV share in %
s_m_l_shares = [0.35, 0.5, 0.15, 0.35, 0.5, 0.15]

bev_phev_shares = {2035: [1/3, 1/3, 1/3, 2/3, 2/3, 2/3],
                   2050: [1/5, 1/5, 1/5, 4/5, 4/5, 4/5]}

battery_capa = [{2035: [12.8, 18.4, 24.0, 56.0, 100.0, 140.0],
                 2050: [16.6, 23.9, 31.2, 72.8, 130.0, 182.0]},
                {2035: [16, 23, 30, 70, 125, 175],
                 2050: [20.8, 29.9, 39.0, 91.0, 162.5, 227.5]},
                {2035: [19.2, 27.6, 36.0, 84.0, 150.0, 210.0, ],
                 2050: [25.0, 35.9, 46.8, 109.2, 195.0, 273.0, ]}]

energy_usage = [{2035: [12.80, 16.00, 19.20, 12.80, 16.00, 19.20],
                 2050: [12.80, 16.00, 19.20, 12.80, 16.00, 19.20]},
                {2035: [16.00, 20.00, 24.00, 16.00, 20.00, 24.00],
                 2050: [16.00, 20.00, 24.00, 16.00, 20.00, 24.00]},
                {2035: [19.20, 24.00, 28.80, 19.20, 24.00, 28.80],
                 2050: [19.20, 24.00, 28.80, 19.20, 24.00, 28.80]}]
efficiency = 0.96  # [%]
efficiency_grid = 0.94  # [%]
soc_min, soc_max = 0.2, 1.0  # [%]
zweck_per_area = {'residential': [6, 7, 8, 9, 10],
                  'retail': [1, 2, 3, 4, 5],
                  'industrial': [1, 2, 8],
                  'agricultural': [1, 2, 8]}

# Size shares: small 35%, mid 50%, large 15%
#   https://www.kba.de/DE/Statistik/Fahrzeuge/Bestand/Motorisierung/b_motorisierung_pkw_dusl.html?nn=652416
# PHEV/BEV shares, siehe MA Tobias Meyr: BEV: 85% PHEV: 15%
charger_availability = 1
average_charging = True

# Optimization variables
save_results = True  # Set to True for runs
daytime_dependency = True
interpolate = True
number_of_tests = 10  # Total minimum number of tests to be evaluated, set to at least 8
number_of_sets = 10  # Total number of sets that are evaluated together to make up the number_of_tests
steps = 24 / t_periods
daytime_split = ['9pm-3am', '3am-9am', '9am-3pm', '3pm-9pm']
daytime_split_t = [0, 6 / steps, 12 / steps, 18 / steps]
max_episodes = 2000  # Number of episodes of CMA-ES, Set to 2000 for runs
lf_iterations = 20  # Maximum number of times each while loop in grid.reinforce is conducted
popsize = 28  # Set to 28 for runs
future = 10
std_deviation = 0.2  # CMA standard deviation, evtl. noch .3
clusters_available = [{2035: 27, 2050: 27},
                      {2035: 36, 2050: 39},
                      {2035: 32, 2050: 35},
                      {2035: 36, 2050: 33},
                      {2035: 27, 2050: 27}]
rounding = 5  # Rounding to x number of digits
tries_if_memoryerror = 3

initial = {'smart_cc_share': .5, 'tc_0': .5, 'tc_1': .5, 'td_0': .5, 'td_1': .5, 'tr_0': .5, 'tv_0': 0, 'tv_1': 0}
bounds = pd.DataFrame.from_dict(
    {'smart_cc_share': [0, 1], 'tc_0': [0, 1], 'tc_1': [0, 1], 'td_0': [0, 1], 'td_1': [0, 1], 'tr_0': [0, 1],
     'tv_0': [-1, 1], 'tv_1': [-1, 1]}).T
theta_names = list(initial.keys())

best_cost_columns = ['grid', 'type', 'year', 'day', 'pev_share', 'p_charge', 'clusters',
                     'investments', 'annuities_grid', 'annuities_charge',
                     'cost_communication', 'grid_losses', 'cost_curtail',
                     'cost_charge_loss', 'cost_battery_degr', 'share_lines',
                     'share_transformers', 'total_cost', 'notes']

run_id = None
parent_folder = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
aws_batch = False  # Set to True if running on AWS


def annuity_factor(cost_type):
    interest_rate = interest[cost_type]
    real_interest = (interest_rate - inflation) / (1 + inflation)
    invest_period_years = invest_years[cost_type]

    af = ((((1 + real_interest) ** invest_period_years) * real_interest) /
          (((1 + real_interest) ** invest_period_years) - 1))  # annuity factor

    return af


def timestep_samples(inputs, grid, elbow_plot=False, visualize=True, start=2, stop=1000, steps=100):
    import numpy as np
    from Functions import cluster_functions as clf

    if inputs.cluster_samples:
        sample = clf.cluster_timesteps_timesteps_weights(grid,
                                                         inputs.sample_size,
                                                         inputs=inputs,
                                                         elbow_plot=elbow_plot,
                                                         visualize=visualize, start=start, stop=stop, steps=steps)
    else:
        # np.random.seed(inputs.edisgo_grid)
        sample = pd.to_datetime(np.sort(np.random.choice(inputs.index[:-3],
                                                         size=inputs.sample_size,
                                                         replace=False)))
        # Build the same DataFrame as from cluster sample but with same weights for all timesteps
        sample = pd.DataFrame(0, index=sample, columns=['clusters', 'weight'])
        sample.loc[:, 'clusters'] = list(range(0, len(sample.index)))
        sample.loc[:, 'weight'] = [len(inputs.index) / len(sample.index)] * len(sample.index)
    return sample


def try_remove_file(filename):
    try:
        os.remove(filename)
    except FileNotFoundError as e:
        logger.error('!!!!!!! File ...{} could not be removed due to: {}'.format(filename[-50:], e))


def aws_upload_results(name, grid_data=False, temp=False, file=None):
    if aws_batch:
        import subprocess

        logger.warning('------------------------Upload------------------------')
        out_file = '{}/Outputs/{}_aws_uploads.txt'.format(parent_folder, name)
        calls = []
        with open(out_file, 'w') as f:
            if file is None:
                if grid_data:
                    calls.append(["aws", "s3", "cp", "./Grids", "s3://chgriddata", "--recursive"])

                if temp:
                    calls.append(["aws", "s3", "rm", "s3://chtemp/{}".format(name), "--recursive"])
                    calls.append(["aws", "s3", "cp", "./Outputs", "s3://chtemp/{}".format(name), "--recursive"])
                else:
                    calls.append(["aws", "s3", "cp", "./Outputs", "s3://choutputs", "--recursive"])
            else:
                if grid_data:
                    calls.append(["aws", "s3", "cp", "{}".format(file), "s3://chgriddata/"])
                elif temp:
                    calls.append(["aws", "s3", "cp", "{}".format(file), "s3://chtemp/{}/".format(name)])
                else:
                    calls.append(["aws", "s3", "cp", "{}".format(file), "s3://choutputs/"])

            calls.append(["aws", "s3", "cp", "nohup.out", "s3://chtemp/{}/".format(name)])

            for call in calls:
                logger.warning('Call: {}'.format(call))
                subprocess.call(call, stdout=f)


def set_design(palette_size=11, fig_size=(22, 12), font_size=10):
    import seaborn as sns
    import matplotlib as mpl

    sns.set_style('whitegrid')
    sns.set(color_codes=True)
    sns.set_palette('colorblind')
    sns.color_palette("husl", palette_size)
    params = {'lines.linewidth': 1,
              'font.size': font_size,
              'font.serif': 'Times New Roman',
              'legend.fontsize': 'x-large',
              'figure.figsize': fig_size,  # Widt, height
              'figure.subplot.hspace': 0,
              'figure.subplot.wspace': 0,
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'ytick.labelsize': 'x-large',
              'axes.xmargin': 0}
    mpl.rcParams.update(params)


def set_logging():
    import warnings
    import logging  # https://docs.python.org/2/howto/logging.html

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("keyring.backend").setLevel(logging.WARNING)
    logging.getLogger("ding0").setLevel(logging.WARNING)
    logging.getLogger("pypsa.pf").setLevel(logging.ERROR)
    logging.getLogger("shapely").setLevel(logging.ERROR)
    logging.getLogger("edisgo").setLevel(logging.WARNING)
    logging.getLogger("base").setLevel(logging.WARNING)
    logging.getLogger("__init__").setLevel(logging.WARNING)
    logging.getLogger("subprocess.call").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")


def creat_batch_files():
    """ Use to generate multiple batch files for various grids

    Need to run the following command on folder with batch files before batching:
    ### find . -type f -print0 | xargs -0 dos2unix ####
    """
    # import pandas as pd

    selected_grids = [1944, 1791, 163, 404, 1055, 1392, 2948, 55]
    # [2620, 1609, 2877, 2063, 1350, 1459, 2743, 1036, 2388, 3335, 2057]
    # [3490, 3357, 798, 1350, 1079, 3383, 194, 2277, 2695, 2330, 1608]
    max_parallel = 28
    for grid in selected_grids:
        text = [
            '#!/bin/bash',
            '#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn69su/pn69su-dss-0000/Logger_files/%j.%x.out',
            '#SBATCH -D ./',
            '#SBATCH -J {}'.format(grid),
            '#SBATCH --get-user-env',
            '#SBATCH --cluster=cm2_tiny',
            '#SBATCH --partition=cm2_tiny',
            '#SBATCH --nodes={}'.format(int(28 / max_parallel)),
            '#SBATCH --ntasks-per-node={}'.format(max_parallel),
            '#SBATCH --mail-type=end',
            '#SBATCH --mail-user=christoph.heilmann@tum.de',
            '#SBATCH --export=NONE',
            '#SBATCH --time=3-00:00:00',
            'cd ..',
            'source myenv4/bin/activate',
            'export LANGUAGE=en_US.UTF-8',
            'export LC_ALL=en_US.UTF-8',
            'export LANG=en_US.UTF-8',
            'export LC_TYPE=en_US.UTF-8',
            'python opt_{}.py'.format(grid),
        ]
        file = 'Batch_files/{}.cmd'.format(grid)
        pd.DataFrame(text).loc[:, 0].to_csv(file, index=False)

        print(grid)
        with open('Run_files/run_file.py') as infile, open('Run_files/opt_{}.py'.format(grid), 'w') as outfile:
            for line in infile:
                if 'max_parallel = 30' in line:
                    line = line.replace('max_parallel = 30',
                                        'max_parallel = {}'.format(max_parallel))
                    print(line)
                if 'edisgo_grid = ' in line:
                    line = line.replace('edisgo_grid = 55',
                                        'edisgo_grid = {}'.format(grid))
                    print(line)
                if '# TODO' in line:
                    continue
                outfile.write(line)


def duplicate_for_tests():
    import re
    from glob import glob
    from shutil import copyfile

    folder = 'C:\\Users\\Christoph Heilmann\\Box Sync\\00. Promotion TUM\\07 Paper No2\\03 MyModel\\Outputs\\' \
             '200510 New results\\New test runs\\*'
    for file in glob(folder):
        splitters = re.split('_', file)
        for nr in range(1, 5):
            name = ''
            position = 0
            for splitter in splitters:
                if position == 0:
                    name = '{}'.format(splitter)
                elif position == 2:
                    name = '{}_{}_{}'.format(name, nr, splitter)
                else:
                    name = '{}_{}'.format(name, splitter)
                position += 1
            copyfile(file, name)

# def do_profile(follow=[]):
#     from line_profiler import LineProfiler
#
#     def inner(func):
#         def profiled_func(*args, **kwargs):
#             try:
#                 profiler = LineProfiler()
#                 profiler.add_function(func)
#                 for f in follow:
#                     profiler.add_function(f)
#                 profiler.enable_by_count()
#                 return func(*args, **kwargs)
#             finally:
#                 profiler.print_stats()
#         return profiled_func
#     return inner
# @do_profile(follow=[])