import dill
import pandas as pd
from datetime import datetime
from Functions import config_functions as cfg
import logging  # https://docs.python.org/2/howto/logging.html
logger = logging.getLogger(__name__)


class Inputs:  # Class for all input variables
    __slots__ = ('edisgo_grid', 'year', 'date', 'weekday', 'day_number', 'index', 'dayindex', 'dates_represented',
                 'dates_represented_index', 'dates_represented_index_array', 'pev_share', 'charging_tech',
                 'p_charge', 'p_discharge', 'safety_margin', 'optimize', 'daytime_dependency', 'curtailment',
                 'interpolate', 'parallel_processing', 'get_solution', 'downsample', 'selected_steps', 'pev_sorting',
                 'episode', 'process_id', 'clusters', 'theta_names_dropped', 'initial_theta',
                 'optimization_thetas', 'name', 'bounds', 'pev_file', 'out_file', 'grid_file', 'grid_file_scenario',
                 'pev_file_test', 'sample_size', 'cluster_samples', 'extreme_cost', 'extreme_file', 'scenario_cc_cost',
                 'scenario_energy_cost', 'scenario_battery', 'scenario_energyuse', 'res_share', 'approach',
                 'socinit_min', 'socinit_max')

    def __init__(self, edisgo_grid=1336, scenario=1, year=2035, charge_load=22, charging_tech='V2G', optimization=True,
                 filename='test', pev_sorting=True, scenario_cc_cost=1, scenario_energy_cost=1,
                 scenario_battery=1, scenario_energyuse=1, res_scenario=1, initial=cfg.initial,
                 socinit_min=0, socinit_max=1):
        import copy

        # Grid variables
        self.edisgo_grid = edisgo_grid
        self.res_share = cfg.res_scenario[res_scenario]

        # Time variables
        self.year = year
        self.date = '2011-01-01'
        self.weekday = datetime.strptime(self.date, '%Y-%m-%d').weekday() + 1  # +1 to move to 1-based indexing
        self.day_number = int(datetime.strptime(self.date, '%Y-%m-%d').strftime('%j')) - 1
        self.index = pd.date_range('2011-01-01',
                                   periods=(365 * cfg.t_periods),
                                   freq=str(24 / cfg.t_periods) + 'H')
        self.dayindex = pd.date_range('2011-01-01', periods=365, freq='d')
        self.dates_represented = None
        self.dates_represented_index = None
        self.dates_represented_index_array = None

        # PEV variables
        self.pev_share = cfg.scenarios[scenario]
        self.charging_tech = charging_tech  # UCC, UCC_delayed, G2V, V2G
        self.p_charge, self.p_discharge = charge_load, -charge_load  # [kW]
        self.safety_margin = 1.6
        self.scenario_cc_cost = scenario_cc_cost
        self.scenario_energy_cost = scenario_energy_cost
        self.scenario_battery = scenario_battery
        self.scenario_energyuse = scenario_energyuse
        self.socinit_min, self.socinit_max = socinit_min, socinit_max

        # Optimization variables
        self.optimize = False
        self.daytime_dependency = True
        self.curtailment = True
        self.interpolate = True
        self.parallel_processing = False
        self.get_solution = False
        self.selected_steps = True
        self.approach = None
        self.pev_sorting = pev_sorting
        self.episode = 0
        self.process_id = 0
        self.sample_size = 100  # Number of sample days used for calculation of grid losses
        self.cluster_samples = True  # Boolean if sample days for losses are clustered or not

        # False, float (take every x-th timestep) or integer (randomly select X dates - int(len(self.index) / 4))
        self.clusters = 'fullyear'  # cfg.clusters_available[scenario][self.year]
        self.theta_names_dropped = []
        self.initial_theta = pd.DataFrame(
            columns=pd.MultiIndex.from_product(
                [cfg.daytime_split, cfg.theta_names]) if self.daytime_dependency is True else cfg.theta_names,
            index=[0]).sort_index(axis=1)
        for theta in cfg.theta_names:
            self.initial_theta.loc[:, pd.IndexSlice[:, theta] if self.daytime_dependency else theta] = initial[theta]

        if self.charging_tech == 'UCC' or self.charging_tech == 'UCC_delayed':
            self.theta_names_dropped.extend(['smart_cc_share', 'tv_0', 'tv_1', 'tc_0', 'tc_1', 'td_0', 'td_1', 'tr_0'])
        if self.charging_tech == 'G2V':
            self.theta_names_dropped.extend(['td_0', 'td_1'])
        if not self.curtailment:
            self.theta_names_dropped.extend(['tr_0'])
            self.initial_theta.loc[:, :] = 0

        if optimization:  # Selected optimization:
            if self.pev_sorting:
                self.theta_names_dropped.extend([])
            else:
                self.theta_names_dropped.extend(['tv_0', 'tv_1'])
        else:  # No optimization:
            self.theta_names_dropped.extend(['smart_cc_share', 'tc_0', 'tc_1', 'td_0', 'td_1', 'tv_0', 'tv_1', 'tr_0'])
        self.theta_names_dropped = list(set(self.theta_names_dropped))  # Remove unintential dublicates
        self.initial_theta = \
            (self.initial_theta.drop(self.theta_names_dropped, axis=1, level=1) if self.daytime_dependency else
             self.initial_theta.drop(self.theta_names_dropped, axis=1))
        self.optimization_thetas = list(cfg.initial.keys())
        for theta in self.theta_names_dropped:
            self.optimization_thetas.remove(theta)
        self.bounds = copy.deepcopy(cfg.bounds)
        self.bounds.set_axis(axis=1, labels=['lower', 'upper'])
        self.bounds.drop(self.theta_names_dropped, inplace=True)
        if str(self.edisgo_grid) in filename:
            self.name = filename
        else:
            self.name = '{}_{}'.format(filename, self.edisgo_grid)

        self.out_file = '{}/Outputs/{}'.format(cfg.parent_folder, self.name)
        self.grid_file = '{}/Grids/Grid_[{}]_gridobject_t{}_y{}_RES{}.pkl'.format(
            cfg.parent_folder, self.edisgo_grid, 17520, self.year, True)
        self.grid_file_scenario = '{}/Grids/Grid_[{}]_gridscenario_y{}_RES{}_{}.pkl'.format(
            cfg.parent_folder, self.edisgo_grid, self.year, True, self.episode)
        self.pev_file = '{}/Grids/PEV_[{}]_pevsobject_p{}_y{}{}{}_{}.pkl'.format(
            cfg.parent_folder, self.edisgo_grid, self.pev_share[self.year], self.year,
            '' if self.scenario_battery == 1 else '_batt{}'.format(self.scenario_battery),
            '' if self.scenario_energyuse == 1 else '_ener{}'.format(self.scenario_energyuse), self.episode)
        self.extreme_cost = {}  # Filename of worst iteration of the base UCC test
        self.extreme_file = {}  # Filename of the best iteration from the G2V optimization

    def update_day(self, day):
        self.date = day
        self.weekday = datetime.strptime(self.date, '%Y-%m-%d').weekday() + 1  # +1 to move to 1-based indexing
        self.day_number = int(datetime.strptime(self.date, '%Y-%m-%d').strftime('%j')) - 1

    def update_names(self):
        self.out_file = '{}/Outputs/{}'.format(cfg.parent_folder, self.name)
        self.grid_file = '{}/Grids/Grid_[{}]_gridobject_t{}_y{}_RES{}.pkl'.format(
            cfg.parent_folder, self.edisgo_grid, 17520, self.year, True)
        self.grid_file_scenario = '{}/Grids/Grid_[{}]_gridscenario_y{}_RES{}_{}.pkl'.format(
            cfg.parent_folder, self.edisgo_grid, self.year, True, self.episode)
        self.pev_file = '{}/Grids/PEV_[{}]_pevsobject_p{}_y{}{}{}_{}_soc{}{}.pkl'.format(
            cfg.parent_folder, self.edisgo_grid, self.pev_share[self.year], self.year,
            '' if self.scenario_battery == 1 else '_batt{}'.format(self.scenario_battery),
            '' if self.scenario_energyuse == 1 else '_ener{}'.format(self.scenario_energyuse), self.episode,
            self.socinit_min, self.socinit_max)

    def get_clusters(self, cluster_file):
        with open(cluster_file, 'rb') as f:
            self.dates_represented = dill.load(f)[0].weight
            logger.warning('=== Getting clusters from {}'.format(cluster_file))
        self.dates_represented_index = pd.date_range(self.dates_represented.index[0],
                                                     periods=cfg.t_periods,
                                                     freq=(str(24 / cfg.t_periods) + 'H'))
        for day in self.dates_represented.index[1:]:
            self.dates_represented_index = self.dates_represented_index.append(
                pd.date_range(day,
                              periods=cfg.t_periods,
                              freq=(str(24 / cfg.t_periods) + 'H')))
        self.date = self.dates_represented.index[0].strftime('%Y-%m-%d')
        self.weekday = datetime.strptime(self.date, '%Y-%m-%d').weekday() + 1  # +1 to move to 1-based indexing
        timedict = dict(zip(self.index, range(len(self.index))))
        self.dates_represented_index_array = [timedict[t] for t in self.dates_represented_index]

    # def downsample_timesteps(self, timesteps):
    #     import numpy as np
    #
    #     try:  # Works if inputs.downsample is an integer
    #         np.random.seed(self.edisgo_grid)
    #         timesteps = pd.to_datetime(np.sort(np.random.choice(timesteps, size=self.downsample, replace=False)))
    #     except TypeError:
    #         pass
    #         if self.downsample is not False:  # Downsample to every second step (hourly)
    #             timesteps = timesteps[::int(1 / self.downsample)]
    #         else:
    #             pass
    #
    #     return timesteps

    def update_worst_best_files(self, return_base=False, test_number=None, combined_expansion=False,
                                invest_not_total=False, fixed=None, new=False):
        import re
        from glob import glob

        cost_best = pd.read_csv('{}_cost_best.csv'.format(self.out_file), index_col=0)

        # Filename of worst iteration of the base UCC test
        cost_base = cost_best.query('type == "base_ucc"')
        if invest_not_total:
            self.extreme_cost['base_ucc'] = cost_base[cost_base.investments == cost_base.investments.max()].iloc[-1]
        else:
            self.extreme_cost['base_ucc'] = cost_base[cost_base.total_cost == cost_base.total_cost.max()].iloc[-1]
        ucc_base_file = glob('{}_base_uccout_e{}_p*_c{}_*'.format(
            self.out_file, self.extreme_cost['base_ucc'].name, int(self.extreme_cost['base_ucc'].total_cost)))[0][:-5]
        self.extreme_file['base_ucc'] = re.split(re.split('_', ucc_base_file)[-1], ucc_base_file)[0]
        if return_base:
            return self.extreme_cost['base_ucc'].name

        # Filename of the best iteration from the G2V optimization
        cost_g2vucc = cost_best.query('type == "g2vucc_fullres"')
        self.extreme_cost['g2vucc_fullres'] = \
            cost_g2vucc[cost_g2vucc.total_cost == cost_g2vucc.total_cost.min()].iloc[-1]
        try:
            g2v_opt_file = glob('{}_out_e{}_p*_c{}_*'.format(
                self.out_file,  self.extreme_cost['g2vucc_fullres'].name,
                self.extreme_cost['g2vucc_fullres'].total_cost))[0][:-5]
        except IndexError:  # Best optimization file has not been save, needs to rerun to be saved
            from Functions import fit_functions as fit

            x0, best_cost = fit.get_best_current_solution(self)  # Get best solution from other runs
            fit.fitness(x0, self, fixed, episode=2222, results_best=9999999, return_full_cost=False, new=new)

            g2v_opt_file = glob('{}_out_e{}_p*_c{}_*'.format(
                self.out_file,  self.extreme_cost['g2vucc_fullres'].name,
                self.extreme_cost['g2vucc_fullres'].total_cost))[0][:-5]

        self.extreme_file['g2vucc_fullres'] = re.split(re.split('_', g2v_opt_file)[-1], g2v_opt_file)[0]

        if test_number is not None:
            if test_number > 0:
                try:  # Filename of worst iteration of the UCC test and of the G2V test
                    for test in ['test_best_g2vucc', 'test_ucc']:
                        cost_test = cost_best.query('type == "{}"'.format(test))
                        self.extreme_cost[test] = cost_test[cost_test.total_cost == cost_test.total_cost.max()].iloc[-1]
                        if combined_expansion:
                            logger.info('Looking for the following files: {}'.format(
                                '{}_{}_t{}_p*_c*_*'.format(self.out_file, test, test_number - 1)))
                            test_file = glob('{}_{}_t{}_p*_c*_*'.format(self.out_file, test, test_number - 1))[0][:-5]
                        else:
                            logger.info('Looking for the following files: {}'.format('{}_{}out_e{}_p*_c*'.format(
                                self.out_file, test, 9000 + test_number - 1)))
                            test_file = glob('{}_{}out_e{}_p*_c*'.format(
                                self.out_file, test, 9000 + test_number - 1))[0][:-5]
                        self.extreme_file[test] = re.split(re.split('_', test_file)[-1], test_file)[0]
                except ValueError:  # If not iteration has been calculated yet
                    pass

    def get_inputs_dict(self, charging_tech, episode, number_of_sets=cfg.number_of_tests):
        import copy

        inputs_dict = {}
        self.charging_tech = charging_tech
        if charging_tech == 'UCC':
            self.curtailment = False
            self.optimize = False
        for test_id in range(0, number_of_sets):
            inputs_dict[test_id] = copy.deepcopy(self)
            inputs_dict[test_id].episode = episode + test_id
            inputs_dict[test_id].update_names()

        return inputs_dict
