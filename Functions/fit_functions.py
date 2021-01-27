import os
import re
import time
import copy
import dill
from glob import glob
import pandas as pd
import numpy as np
from datetime import datetime
from Functions import \
    config_functions as cfg
import logging

logger = logging.getLogger(__name__)
cfg.set_logging()


class Fitness:
    __slots__ = ('x', 'results_best', 'theta', 'thetas', 'cost_episode', 'energy_charge', 'equipment_exp',
                 'theta_track', 'time_track', 'energy_charge', 'energy_discharged', 'energy_curtailed',
                 'return_full_cost')

    def __init__(self, x, results_best, return_full_cost=True):
        """
        Object for all optimization variables and fit functions
        """

        self.x = x
        self.results_best = results_best
        self.theta = pd.DataFrame()
        self.thetas = pd.DataFrame()

        index = pd.MultiIndex(levels=[[], []],
                              labels=[[], []],
                              names=['episode', 'process_id'])
        self.cost_episode = pd.DataFrame(index=index,
                                         columns=cfg.cost_types)
        self.energy_charge = pd.DataFrame(index=index,
                                          columns=['energy_need', 'energy_charged', 'energy_must', 'energy_smart',
                                                   'energy_avrg', 'energy_discharged', 'energy_curtailed'])
        self.equipment_exp = pd.DataFrame(index=index,
                                          columns=['share_lines', 'share_transformers'])
        self.time_track = pd.DataFrame(index=index,
                                       columns=['start_episode', 'prep_episode', 'pev_curt_calc', 'cost_calc',
                                                'save_results'])
        self.theta_track = pd.DataFrame(index=index,
                                        columns=pd.MultiIndex.from_product([cfg.daytime_split, cfg.theta_names]))
        self.energy_discharged = 0
        self.energy_curtailed = 0
        self.return_full_cost = return_full_cost

    def save_blank_vars(self, inputs):
        for variable, name in list(zip(
                [self.cost_episode, self.energy_charge, self.equipment_exp, self.theta_track, self.time_track],
                ['cost', 'energy_charge', 'equipment', 'thetas', 'runtime'])):
            file = '{}/Outputs/{}_{}.csv'.format(cfg.parent_folder, inputs.name, name)
            if not os.path.isfile(file):
                variable.to_csv(file)  # Set initial CSV file

    def prepare_thetas(self, inputs, fixed):
        if fixed is not None:
            for t in fixed.keys():
                try:
                    inputs.optimization_thetas.remove(t)
                    inputs.theta_names_dropped.append(t)
                except ValueError:
                    pass

        index = pd.MultiIndex.from_arrays([[inputs.episode], [inputs.process_id]], names=('episode', 'process_id'))
        columns = (pd.MultiIndex.from_product([cfg.daytime_split, inputs.optimization_thetas]) if
                   inputs.daytime_dependency else inputs.optimization_thetas)

        if 'smart_cc_share' in inputs.optimization_thetas:
            for position in np.where(columns.labels[1] == np.where(columns.levels[1] == 'smart_cc_share'))[1]:
                self.x = np.insert(self.x, position, self.x[-1])
            self.x = self.x[:-1]

        opt_thetas = pd.DataFrame(self.x if inputs.optimization_thetas != [] else [],
                                  index=columns,
                                  columns=index).T
        fixed_thetas = pd.DataFrame(0,
                                    index=index,
                                    columns=(pd.MultiIndex.from_product([cfg.daytime_split, inputs.theta_names_dropped])
                                             if inputs.daytime_dependency else inputs.theta_names_dropped)
                                    ).sort_index(1)

        if not any(item in ['tv_0', 'tv_1'] for item in inputs.optimization_thetas):
            fixed_thetas.loc[:, pd.IndexSlice[:, 'tv_0']] = 1  # To have at least one sorting theta
        if fixed is not None:
            for theta in fixed.keys():
                if theta not in inputs.optimization_thetas:
                    fixed_thetas.loc[:, pd.IndexSlice[:, theta]] = fixed[theta]

        self.theta = pd.concat([opt_thetas, fixed_thetas], axis=1).sort_index(1)
        self.thetas = pd.DataFrame(index=range(cfg.t_periods),
                                   columns=cfg.theta_names,
                                   dtype='float32')
        if inputs.daytime_dependency:
            for t, time_period in zip(cfg.daytime_split_t, cfg.daytime_split):
                self.thetas.loc[t, :] = self.theta.loc[(inputs.episode, inputs.process_id), time_period]
            self.thetas = self.thetas.append(self.thetas).append(self.thetas).reset_index(drop=True)
            self.thetas = self.thetas.interpolate(method='cubic', axis=0, inplace=False)

            # Keep interpolation within limits of thetas
            self.thetas[self.thetas < inputs.bounds.lower] = (self.thetas < inputs.bounds.lower) * inputs.bounds.lower
            self.thetas[self.thetas > inputs.bounds.upper] = (self.thetas > inputs.bounds.upper) * inputs.bounds.upper
            self.thetas = np.asarray(np.rec.fromrecords(self.thetas.iloc[cfg.t_periods:(2 * cfg.t_periods), :],
                                                        names=self.thetas.columns.tolist()))
        else:
            self.thetas.loc[0, :] = self.theta.loc[(inputs.episode, inputs.process_id), '9pm-3am']
            self.thetas = self.thetas.ffill()
            self.thetas = np.asarray(np.rec.fromrecords(self.thetas,
                                                        names=self.thetas.columns.tolist()))

    def fitness_run(self, inputs, name='g2vucc_fullres', test_number=None, combined_expansion=False,
                    original_grid=True, new=False, new_avrg_cc=False):
        from Functions import \
            grid_functions as grd, \
            cost_functions as csf, \
            pev_functions as pev

        self.time_track.loc[(inputs.episode, inputs.process_id), 'start_episode'] = datetime.now()
        start_d = time.time()
        logger.debug('Calculating PEV charging for grid {} year {}, process id: {}'.format(
            inputs.edisgo_grid, inputs.year, os.getpid()))

        # Get grid: get a new, initial grid for all optimization runs and test UCC runs & get an optimized grid
        # for all G2VUCC test runs
        grid = grd.get_optimized_grid(inputs,
                                      test_number=test_number,
                                      name=name,
                                      original_grid=original_grid,
                                      combined_expansion=combined_expansion)
        self.time_track.loc[(inputs.episode, inputs.process_id), 'prep_episode'] = round(time.time() - start_d, 2)

        start_d = time.time()
        pevs = pev.get_pev(inputs,
                           grid=grid,
                           smart_cc_share=self.thetas['smart_cc_share'].mean())  # Get PEV object
        pevs.calc_charging_curtailment(grid, inputs, self.thetas, new=new, new_avrg_cc=new_avrg_cc)
        logger.debug('PEV charging and curtailment calculated for grid {} year {}, process id: {}'.format(
            inputs.edisgo_grid, inputs.year, os.getpid()))
        pevs = grid.change_demand_loads(pevs, inputs, test_number)  # Add PEV to demand

        # Add curtailment to generation
        curtailed_per_unit = grid.add_curtailment(inputs) if inputs.curtailment else None
        logger.debug('Demand and generation loads changed for grid {} year {}, process id: {}'.format(
            inputs.edisgo_grid, inputs.year, os.getpid()))
        self.time_track.loc[(inputs.episode, inputs.process_id), 'pev_curt_calc'] = round(time.time() - start_d, 2)

        start_d = time.time()
        cost = csf.CostCalc(inputs)  # Initialize cost variable
        self.energy_discharged, self.energy_curtailed, grid = \
            cost.calc_all_cost(grid, pevs, inputs,
                               curtailed_per_unit=curtailed_per_unit,
                               name=name,
                               combined_expansion=combined_expansion)

        self.cost_episode.loc[(inputs.episode, inputs.process_id), :] = cost.cost  # Save cost of current day
        if test_number is None:  # Tracking equipment changes
            self.equipment_exp.loc[(inputs.episode, inputs.process_id), :] = grid.get_equipment_expansion()
            logger.debug('Cost calculation done for grid {} year {}'.format(inputs.edisgo_grid, inputs.year))
        self.time_track.loc[(inputs.episode, inputs.process_id), 'cost_calc'] = round(time.time() - start_d, 2)

        return grid, cost, pevs

    def save_episode_results(self, inputs, cost, grid, pevs, name='g2vucc_fullres', test_number=None,
                             original_grid=False, save_results=cfg.save_results, combined_expansion=False):
        start_d = time.time()
        # For testing: add grid expansions from previous iterations
        add_investments, add_annities_grid, add_share_lines, add_share_transformers = 0, 0, 0, 0

        if (test_number is not None) and name != 'base_ucc':
            if combined_expansion:
                grid.get_reinforcement_timesteps(inputs, pevs.ppev_load, test_run=(test_number is not None))
                temp_file = '{}_{}_e{}_c{}.dill'.format(
                    inputs.out_file, name, inputs.episode, cost.cost.sum())
                with open(temp_file, 'wb') as f:
                    dill.dump([grid.get_loads_dict(getting=True),
                               grid.timesteps,
                               {'rides_at_node': pevs.rides_at_node,
                                'ppev_load': pevs.ppev_load,
                                'ppev_load_must': pevs.ppev_load_must,
                                'original_load_delta': pevs.original_load_delta,
                                'remaining_load_delta': pevs.remaining_load_delta,
                                'fleet_cc_capa': pevs.fleet_cc_capa,
                                'fleet_dc_capa': pevs.fleet_dc_capa}], f)
            else:
                if test_number == 0:  # Only do this if current test is after optimization
                    if original_grid:  # Add no previous cost if original grid is used
                        cost_previous = pd.Series(0, index=['investments', 'annuities_grid',
                                                            'share_lines', 'share_transformers'])
                    else:  # Add cost from best run of optimization
                        if name == 'test_best_g2vucc':
                            cost_previous = inputs.extreme_cost['g2vucc_fullres']
                        elif name == 'test_ucc':
                            cost_previous = inputs.extreme_cost['base_ucc']
                        else:
                            cost_previous = None
                else:  # Add previous expansion cost if available
                    cost_best = pd.read_csv('{}_cost_best.csv'.format(inputs.out_file), index_col=0)
                    cost_previous = cost_best.query('type == "{}"'.format(name)).loc[inputs.episode - 1, :]

                add_investments += cost_previous.loc['investments']
                add_annities_grid += cost_previous.loc['annuities_grid']
                add_share_lines += cost_previous.loc['share_lines']
                add_share_transformers += cost_previous.loc['share_transformers']

        if (cost.cost.sum() <= self.results_best) or (test_number is not None):
            if not self.return_full_cost:
                # Get the expanded equipment share of the original equipment in the grid
                expanded_equipment = grid.edisgo.network.results.grid_expansion_costs
                share_lines, share_transformers = 0, 0
                if expanded_equipment is not None:
                    share_lines, share_transformers = grid.get_equipment_expansion()
                grid_problems, save_results = grid.get_grid_problems(save_results)

                edisgo_grid_name = '{}{}'.format(inputs.edisgo_grid, '' if inputs.pev_sorting else 'nosorting')
                cost_all = pd.DataFrame(index=[], columns=cfg.best_cost_columns)

                # Save output to the overall csv file
                cost_all.loc[inputs.episode, :] = [
                    edisgo_grid_name, name,
                    inputs.year, 'all', inputs.pev_share[inputs.year], inputs.p_charge, inputs.clusters,
                    cost.investments_grid, cost.cost.annuities_grid, cost.cost.annuities_charge,
                    cost.cost.cost_communication, cost.cost.cost_grid_losses, cost.cost.cost_curtail,
                    cost.cost.cost_charge_loss, cost.cost.cost_battery_degr,
                    share_lines, share_transformers,
                    cost.cost.sum(),
                    'Number of grid problems: {}'.format(grid_problems)]
                if test_number is not None:
                    cost_all.to_csv('{}_cost_best_test.csv'.format(inputs.out_file), mode='a', header=False)
                    cost_all.investments += add_investments
                    cost_all.annuities_grid += add_annities_grid
                    cost_all.share_lines += add_share_lines
                    cost_all.share_transformers += add_share_transformers
                    cost_all.total_cost += add_annities_grid
                if not combined_expansion:
                    cost_all.to_csv('{}_cost_best.csv'.format(inputs.out_file), mode='a', header=False)

            # Only for optimization and basecost calculation: Save grid, pevs and costfit files
            if (save_results and test_number is None) or name == 'base_ucc':  # Remove all previous files
                self.results_best = cost.cost.sum()  # Save current results as best solution
                if test_number is None:
                    for x in glob('{}_out_*.dill'.format(inputs.out_file)):
                        if int(re.split('_', x)[-4][1:]) < inputs.episode:
                            try:  # Remove if available, could have been already removed by parallel iteration
                                os.remove(x)
                            except FileNotFoundError:
                                pass
                if inputs.episode > 1:  # No need to save all first episode results only to delete later
                    file = '{}_{}out_e{}_p{}_c{}'.format(
                        inputs.out_file, name if (test_number is not None) else '', inputs.episode,
                        os.getpid(), cost.cost.sum())
                    for variable, var_name in zip([grid, pevs, [cost, self]], ['grid', 'pevs', 'costfit']):
                        try:
                            with open('{}_{}.dill'.format(file, var_name), 'wb') as f:
                                dill.dump(variable, f)
                        except (MemoryError, OSError) as e:
                            logger.warning('Could not save {} due to {}'.format(file, e))
                            pass

        # Logging information from current episode
        logger.debug('Current cost for grid {}: \n{}'.format(inputs.edisgo_grid, cost.cost))
        self.energy_charge.loc[(inputs.episode, inputs.process_id), :] = \
            [round(sum(pevs.data['energy_tb_charged']), cfg.rounding),
             round(pevs.ppev_load.sum().sum() * (24 / cfg.t_periods), cfg.rounding),
             round(pevs.ppev_load_must.sum().sum() * (24 / cfg.t_periods), cfg.rounding),
             round(pevs.ppev_load_smart.sum().sum() * (24 / cfg.t_periods), cfg.rounding),
             round(pevs.ppev_load_avrg.sum().sum() * (24 / cfg.t_periods), cfg.rounding),
             round(self.energy_discharged, cfg.rounding),
             round(self.energy_curtailed, cfg.rounding)]  # [kWh]
        self.time_track.loc[(inputs.episode, inputs.process_id), 'save_results'] = round(time.time() - start_d, 2)
        for variable, var_name in zip(
                [self.theta, self.energy_charge, self.cost_episode, self.equipment_exp, self.time_track],
                ['thetas', 'energy_charge', 'cost', 'equipment', 'runtime']):
            variable.to_csv('{}_{}.csv'.format(inputs.out_file, var_name),
                            mode='a',
                            header=False)  # Appending to CSV files

        return cost.cost.sum()


def get_best_current_solution(inputs, test_run=False):
    try:
        if not test_run:
            cma_file = '{}/Outputs/{}_CMA.dill'.format(cfg.parent_folder, inputs.name)
            if os.path.isfile(cma_file):  # New approach
                with open(cma_file, 'rb') as f:
                    es = dill.load(f)
                best_cost = es.best.f
                best_thetas = es.best.x

                logger.warning('=== Current best solution: {} ==='.format(best_cost))
                logger.info('=== Current best thetas: {} ==='.format(best_thetas))

                return best_thetas.tolist(), best_cost  # Only optimization thetas
        else:
            cost = pd.read_csv('{}/Outputs/{}_cost.csv'.format(cfg.parent_folder, inputs.name),
                               index_col=['episode', 'process_id'])
            cost = cost.query("episode <= 7000")  # Only keep optimization runs (> 7000 reserved for testing)

            costfit_file = '{}/Outputs/{}_out_e{}_p{}_c{}_costfit.dill'.format(
                cfg.parent_folder, inputs.name, cost.sum(1).argmin()[0], cost.sum(1).argmin()[1],
                int(cost.sum(1).min()))
            if not os.path.isfile(costfit_file):  # If the same cost has been found in multiple iterations
                files = '{}/Outputs/{}_out_*_c{}_costfit.dill'.format(
                    cfg.parent_folder, inputs.name, int(cost.sum(1).min()))
                costfit_file = glob(files)[0]

            logger.warning('Getting fit object for test run from file'
                           '\n ...{}'.format(costfit_file[-50:]))
            with open(costfit_file, 'rb') as f:
                cost_obj, fit_obj = dill.load(f)

            return fit_obj

    except (IndexError, ValueError, FileNotFoundError) as e:  # No solutions are existing
        logger.error('Error in getting best current solution: {}'.format(e))
        return None, None


def fitness(x, inputs, fixed, episode=0, results_best=9999999, return_full_cost=False, new=False, new_avrg_cc=False):
    import traceback

    start_f = time.time()
    fit = Fitness(x, results_best, return_full_cost)
    episode += 1
    inputs.episode = episode
    inputs.process_id = os.getpid()
    fit.prepare_thetas(inputs, fixed)

    logger.info('STARTING EPISODE {} OF CMA on process {} with thetas: \n{}'.format(
        inputs.episode, inputs.process_id,
        fit.theta.loc[pd.IndexSlice[episode, inputs.process_id], :].mean(level=1, axis=0).replace(0, np.nan).dropna()))
    if inputs.episode < 2:
        fit.save_blank_vars(inputs)
    fit.time_track.loc[(inputs.episode, inputs.process_id), 'prep_episode'] = \
        copy.deepcopy(round(time.time() - start_f, 2))

    try:
        grid, cost, pevs = fit.fitness_run(inputs, new=new, new_avrg_cc=new_avrg_cc)
        result = fit.save_episode_results(inputs, cost, grid, pevs)
        logger.info('Total cost for grid {} in episode {} and process {}: {}'.format(
            inputs.edisgo_grid, inputs.episode, os.getpid(), cost.cost.sum()))

        return result
    except AssertionError as e:
        logger.warning('Error in episode {} and process {}: \n{}: {}\n {}'.format(
            inputs.episode, inputs.process_id, type(e), e, traceback.format_exc()))

        return 999999999


def test_best_solutions(inputs_name, test_number=0, combined_expansion=False, original_grid=True, new=False,
                        new_avrg_cc=False):
    start_f = time.time()
    inputs, name = inputs_name
    inputs.process_id = os.getpid()
    logger.warning('Starting the calculation of {} for grid {} and episode {} on {}'.format(
        name, inputs.edisgo_grid, inputs.episode, inputs.process_id))

    if (name == 'test_ucc') or (name == 'base_ucc'):
        initial_theta = inputs.initial_theta.drop('smart_cc_share', level=1, axis=1).dropna(axis=1).iloc[0].tolist() + [
                    inputs.initial_theta.loc[:, pd.IndexSlice[:, 'smart_cc_share']].mean().mean()]
        fit = Fitness(initial_theta, results_best=9999999, return_full_cost=False)
        fit.thetas = np.zeros(cfg.t_periods,
                              dtype=list(zip(cfg.theta_names,
                                             ['i4'] * len(cfg.theta_names))))  # UCC charging & no curtailment
    else:
        fit = get_best_current_solution(inputs, test_run=True)  # Get best solution from other runs

        # Set tracking variables
        fit.theta.index.set_levels([[inputs.episode], [inputs.process_id]], inplace=True)
        fit.cost_episode.drop(fit.cost_episode.index, inplace=True)
        fit.energy_charge.drop(fit.energy_charge.index, inplace=True)
        fit.equipment_exp.drop(fit.equipment_exp.index, inplace=True)
        fit.time_track.drop(fit.time_track.index, inplace=True)
        fit.theta_track.drop(fit.theta_track.index, inplace=True)

    fit.save_blank_vars(inputs)
    fit.time_track.loc[(inputs.episode, inputs.process_id), 'prep_episode'] = \
        copy.deepcopy(round(time.time() - start_f, 2))

    grid, cost, pevs = fit.fitness_run(inputs,
                                       name=name,
                                       test_number=test_number,
                                       combined_expansion=combined_expansion,
                                       original_grid=original_grid,
                                       new=new,
                                       new_avrg_cc=new_avrg_cc)

    total_cost = fit.save_episode_results(inputs, cost, grid, pevs,
                                          name=name,
                                          test_number=test_number,
                                          original_grid=original_grid,
                                          save_results=True,
                                          combined_expansion=combined_expansion)
    logger.info('### TOTAL COST of the test run for grid {} are {} Euro'.format(
        inputs.edisgo_grid, total_cost))


def test_combined_expansion_prep(test_dict, max_parallel, test_number, original_grid=True, combined_expansion=True,
                                 new=False, new_avrg_cc=False):
    """Test the optimized strategy on a number of PEV sets but evaluate all grid expansion cost together"""
    import multiprocessing as mp
    from functools import partial

    # Check if some solutions are already available
    inputs_not_done = []
    for name in test_dict.keys():
        for inputs in list(test_dict[name].values()):
            try:
                temp_file = glob('{}_{}_e{}_c*.dill'.format(inputs.out_file, name, inputs.episode))[0]
                if not os.path.isfile(temp_file):
                    inputs_not_done.append([inputs, name])
            except IndexError:
                inputs_not_done.append([inputs, name])

    if len(inputs_not_done) > 0:
        pool = mp.Pool(processes=min(len(inputs_not_done), max_parallel),
                       maxtasksperchild=1)
        func = partial(test_best_solutions,
                       test_number=test_number,
                       combined_expansion=combined_expansion,
                       original_grid=original_grid,
                       new=new,
                       new_avrg_cc=new_avrg_cc)
        pool.map(func, inputs_not_done, chunksize=1)
        pool.close()
        pool.join()


def test_combined_expansion_calc(name, test_dict, test_number, original_grid=True):
    from Functions import \
        grid_functions as grd, \
        cost_functions as csf

    inputs_dict = test_dict[name]
    inputs = list(inputs_dict.values())[0]

    # Get empty pypsa.descriptors.Dict
    grid_test = grd.get_grid(inputs)  # Only needed to get empty pypsa.descriptors.Dict
    empty_data = np.empty([inputs.index.size * len(inputs_dict.keys()),
                          len(grid_test.nodes['node'])], dtype='float32')
    combined_loads_dict = {
        'pd_load': copy.deepcopy(empty_data), 'qd_load': copy.deepcopy(empty_data),
        'pp_load': copy.deepcopy(empty_data), 'qp_load': copy.deepcopy(empty_data),
        'pcurtail': copy.deepcopy(empty_data), 'qcurtail': copy.deepcopy(empty_data),
        'loads_t': grid_test.edisgo.network.pypsa.loads_t, 'generators_t': grid_test.edisgo.network.pypsa.generators_t}
    for load in ['p_set', 'q_set', 'p', 'q']:
        combined_loads_dict['loads_t'][load] = pd.DataFrame()
        combined_loads_dict['generators_t'][load] = pd.DataFrame()
    combined_pev_data = {
        'rides_at_node': copy.deepcopy(empty_data),
        'ppev_load': copy.deepcopy(empty_data), 'ppev_load_must': copy.deepcopy(empty_data),
        'original_load_delta': copy.deepcopy(empty_data), 'remaining_load_delta': copy.deepcopy(empty_data),
        'fleet_cc_capa': copy.deepcopy(empty_data), 'fleet_dc_capa': copy.deepcopy(empty_data)}
    combined_timesteps = pd.DatetimeIndex(start='2011-01-01', freq='.5H', periods=0)
    original_timesteps_samples = pd.DatetimeIndex(start='2011-01-01', freq='.5H', periods=0)
    combined_timesteps_samples = pd.DatetimeIndex(start='2011-01-01', freq='.5H', periods=0)

    # Random sample distributed across all sets to be added to combined_timesteps for grid losses
    timestep_samples = np.random.choice(grid_test.random_sample.index,
                                        size=(len(inputs_dict),
                                              int(inputs.sample_size / len(inputs_dict))),
                                        replace=False)

    # Combine loads from different years, last timesteps are taken out because
    # they are not representative if continued application
    for inputs_id, inputs in zip(inputs_dict.keys(), inputs_dict.values()):
        temp_file = glob('{}_{}_e{}_c*.dill'.format(inputs.out_file, name, inputs.episode))[0]
        with open(temp_file, 'rb') as f:
            loads_dict, timesteps, pev_data = dill.load(f)

        # Add sample timesteps for grid losses to timesteps for grid expansion
        timesteps_original = timesteps['initial'].append(
            pd.DatetimeIndex(timestep_samples[inputs_id])).unique().sort_values()
        offset = pd.offsets.DateOffset(year=2011 + inputs_id)
        timesteps['initial'] = timesteps_original + offset  # Offset year for new scenario
        combined_timesteps = combined_timesteps.append(timesteps['initial'])  # Add timesteps of set to all timesteps
        # Track timestep samples to make sure that combined grid is aligned
        original_timesteps_samples = original_timesteps_samples.append(pd.DatetimeIndex(timestep_samples[inputs_id]))
        combined_timesteps_samples = combined_timesteps_samples.append(pd.DatetimeIndex(timestep_samples[inputs_id]) +
                                                                       offset)

        index_from = inputs_id * inputs.index.size
        index_to = index_from + inputs.index.size
        # Gathering PEV data
        for key in combined_pev_data.keys():
            combined_pev_data[key][index_from:index_to, :] = pev_data[key]

        # Gathering node-based load data
        for key in ['pd_load', 'qd_load', 'pp_load', 'qp_load', 'pcurtail', 'qcurtail']:
            if not loads_dict[key].size == 0:  # Exclude curtail if curtailment is not allowed
                combined_loads_dict[key][index_from:index_to, :] = loads_dict[key]
        # Gathering grid data and selecting the relevant timesteps
        for key in ['loads_t', 'generators_t']:
            for load_type in ['p_set', 'q_set', 'p', 'q']:
                loads_dict[key][load_type] = loads_dict[key][load_type].loc[timesteps_original, :]  # Select timesteps
                loads_dict[key][load_type].index = timesteps['initial']  # Change timesteps to appropriate year
                # Add timesteps from set to combined timesteps
                combined_loads_dict[key][load_type] = pd.concat([combined_loads_dict[key][load_type],
                                                                 loads_dict[key][load_type]])

        # # Remove all temp files that were used for the test runs
        # if cfg.remove_test_timeseries:
        #     os.remove(temp_file)

    inputs = copy.deepcopy(list(inputs_dict.values())[0])
    inputs.process_id = os.getpid()
    inputs.index = combined_timesteps
    inputs.episode = 9999 if name != 'test_ucc' else 8888

    # Set combined loads on the grid from the optimization runs
    grid = grd.get_optimized_grid(inputs,
                                  test_number=test_number,
                                  loads_dict=combined_loads_dict,
                                  name=name,
                                  original_grid=original_grid)  # Get grid
    grid.timesteps['initial'] = combined_timesteps
    grid.random_sample = grid_test.random_sample
    grid.random_sample = grid.random_sample.loc[original_timesteps_samples]
    grid.random_sample.index = combined_timesteps_samples

    # Calculate the combined grid expansion cost
    cost = csf.CostCalc(inputs)  # Initialize cost variable
    grid = cost.investment_needs(grid, None, inputs, combined_expansion=True)  # Invest. annuities
    cost.grid_losses(grid, inputs, daily=False)  # Calculate distribution system energy loss cost

    # Save results
    # expanded_equipment = grid.edisgo.network.results.grid_expansion_costs
    investments_grid = round(grid.edisgo.network.results.grid_expansion_costs.sum()['total_costs'] * 1e3,
                             0)  # [kEUR] to [EUR])
    share_lines, share_transformers = grid.get_equipment_expansion()
    grid_problems, _ = grid.get_grid_problems()

    cost_all = pd.DataFrame(index=[], columns=cfg.best_cost_columns)
    cost_best_test = pd.read_csv('{}_cost_best_test.csv'.format(inputs.out_file), index_col=[0]
                                 ).query('type == "{}"'.format(name))
    for inputs in inputs_dict.values():
        # Get expansion cost from current iteration
        cost_best_test_episode = cost_best_test.loc[inputs.episode]
        cost_all.loc[inputs.episode, :] = [
            inputs.edisgo_grid, name,
            inputs.year, 'all', inputs.pev_share[inputs.year], inputs.p_charge, inputs.clusters,
            int(investments_grid), cost.cost.annuities_grid, cost_best_test_episode.annuities_charge,
            cost_best_test_episode.cost_communication, cost.cost.cost_grid_losses, cost_best_test_episode.cost_curtail,
            cost_best_test_episode.cost_charge_loss, cost_best_test_episode.cost_battery_degr,
            share_lines, share_transformers,
            0, 'Number of grid problems: {}'.format(grid_problems)]

        # Get expansion cost from previous iterations
        for cost_type in ['investments', 'annuities_grid', 'share_lines', 'share_transformers']:
            if test_number == 0:  # Only do this if current test is after optimization
                if not original_grid:
                    if name == 'test_ucc':    # Add cost from worst run of base UCC
                        cost_all.loc[inputs.episode, cost_type] += inputs.extreme_cost['base_ucc'].loc[cost_type]
                    if name == 'test_best_g2vucc':    # Add cost from best run of optimization
                        cost_all.loc[inputs.episode, cost_type] += inputs.extreme_cost['g2vucc_fullres'].loc[cost_type]
            else:  # Add previous expansion cost if available
                cost_all.loc[inputs.episode, cost_type] += inputs.extreme_cost[name].loc[cost_type]

        # Sum up the total cost
        cost_all.loc[inputs.episode, 'total_cost'] = \
            cost_all.loc[inputs.episode, 'annuities_grid':'cost_battery_degr'].sum()

    # Save results
    cost_all.to_csv('{}_cost_best.csv'.format(inputs.out_file), mode='a', header=False)

    file = '{}_{}_t{}_p{}_c{}'.format(
        inputs.out_file, name, test_number, os.getpid(), cost_all.loc[:, 'total_cost'].max())
    for variable, name in zip([grid, [cost, None], combined_pev_data], ['grid', 'costfit', 'pevs']):
        out_file = '{}_{}.dill'.format(file, name)
        try:
            with open(out_file, 'wb') as f:
                dill.dump(variable, f)
        except (MemoryError, OSError) as e:
            logger.error('!!!!!!!'
                         '\n File ...{} could not be saved due to:'
                         '\n{}'.format(out_file[-50:], e))
    logger.debug('Cost calculation done for grid {} year {}'.format(inputs.edisgo_grid, inputs.year))

    return cost.cost.annuities_grid


def test_final_losses(inputs_name, test_number, combined_expansion, original_grid, new=False, new_avrg_cc=False):
    from Functions import \
        grid_functions as grd, \
        cost_functions as csf, \
        loadflow_functions as lff

    inputs, name = inputs_name
    temp_file = '{}_{}_e{}_c*.dill'.format(inputs.out_file, name, inputs.episode)
    if len(glob(temp_file)) < 1:
        test_best_solutions(inputs_name=[inputs, name],
                            test_number=test_number,
                            combined_expansion=combined_expansion,
                            original_grid=original_grid,
                            new=new,
                            new_avrg_cc=new_avrg_cc)
    with open(glob(temp_file)[0], 'rb') as f:
        loads_dict, timesteps, pev_data = dill.load(f)

    # Set combined loads on the grid from the optimization runs
    grid = grd.get_optimized_grid(inputs,
                                  test_number=test_number,
                                  loads_dict=loads_dict,
                                  name=name,
                                  original_grid=original_grid,
                                  set_p_set_only=True)  # Get grid

    # Get original grid sample timesteps and analyze grid
    grid_org = grd.get_grid(inputs)
    grid.random_sample.index = grid_org.random_sample.index
    grid.edisgo, _ = lff.analyze(grid.edisgo, mode='mv', timesteps=grid_org.random_sample.index)

    # Calculate the combined grid expansion cost
    cost = csf.CostCalc(inputs)  # Initialize cost variable
    cost.grid_losses(grid, inputs, daily=False)  # Calculate distribution system energy loss cost

    # Save results
    cost_best = pd.read_csv('{}_cost_best.csv'.format(inputs.out_file), index_col=[0]
                            ).query('type == "{}"'.format(name))
    cost_all = pd.DataFrame(index=[], columns=cfg.best_cost_columns)
    cost_all.loc[inputs.episode + 900, :] = cost_best.loc[inputs.episode, :]
    cost_all.loc[inputs.episode + 900, 'grid_losses'] = cost.cost.cost_grid_losses
    cost_all.loc[inputs.episode + 900, 'total_cost'] = \
        cost_all.loc[inputs.episode + 900, 'annuities_grid':'cost_battery_degr'].sum()  # Sum up the total cost
    cost_all.to_csv('{}_cost_best.csv'.format(inputs.out_file), mode='a', header=False)  # Save results
    logger.debug('Cost calculation done for grid {} year {}'.format(inputs.edisgo_grid, inputs.year))


def basecost_calc(inputs, max_parallel, invest_not_total=False, number_of_sets=cfg.number_of_tests, new=False,
                  new_rides=False):
    import multiprocessing as mp
    from functools import partial
    from Functions import \
        pev_functions as pev, \
        grid_functions as grd
    from glob import glob

    logger.warning('\n==================================================================================='
                   '\nStarting basecost UCC test to identify worst set of PEV rides'
                   '\n===================================================================================')
    # Get inputs
    inputs_dict = inputs.get_inputs_dict(charging_tech='UCC',
                                         episode=9000,
                                         number_of_sets=number_of_sets)
    grd.get_grid(inputs, first=True)  # Make sure grid object is available

    # Get load scenarios
    grd.get_set_of_loads(inputs_dict,
                         max_parallel=max_parallel)
    # Get PEV scenarios
    pev.get_set_of_pev(inputs_dict,
                       max_parallel=max_parallel,
                       new_rides=new_rides)

    # Update number of chargers to max across all scenarios
    pev.update_number_chargers(inputs_dict, number_of_sets=number_of_sets)

    cost_best_test = pd.read_csv('{}_cost_best_test.csv'.format(inputs.out_file),
                                 index_col=[0]).query('type == "base_ucc"')
    sets_not_done = []
    for rides_set in inputs_dict.values():
        if rides_set.episode not in cost_best_test.index:
            sets_not_done.append([rides_set, 'base_ucc'])
        else:
            logger.info('Basecost already available for: {}'.format(rides_set.episode))

    if sets_not_done:
        pool = mp.Pool(processes=min(max(1, len(sets_not_done)), max_parallel),
                       maxtasksperchild=1)
        func = partial(test_best_solutions)
        pool.map(func, sets_not_done, chunksize=1)
        pool.close()
        pool.join()

        if cfg.remove_test_timeseries:
            basecost_files = glob('{}/Outputs/{}_base_uccout_e*_p*_c*_*.dill'.format(
                cfg.parent_folder, inputs.name))
            worst_episode = inputs.update_worst_best_files(return_base=True, invest_not_total=invest_not_total, new=new)
            for file in basecost_files:
                if not (str(worst_episode) in file):
                    os.remove(file)

        cfg.aws_upload_results(inputs.name, grid_data=True)


def get_es(inputs, cma_file, fixed, cfg_popsize, number_of_processes, max_episodes):
    import cma

    if os.path.isfile(cma_file):  # Get existing CMA runs if available
        with open(cma_file, 'rb') as f:
            es = dill.load(f)
        # x0 = es.best.x
        es.opts['maxfevals'] = cfg.max_episodes
        max_episodes = cfg.max_episodes
    else:
        x0, best_cost = get_best_current_solution(inputs) if inputs.get_solution \
                            else (None, None)  # Get best solution from other runs

        # Add and remove fixed thetas as needed
        if fixed is not None:
            for var in fixed.keys():
                inputs.optimization_thetas.remove(var)
                inputs.bounds.drop(var, inplace=True)
                inputs.initial_theta.drop(var, axis=1, level=1, inplace=True)
                inputs.theta_names_dropped.append(var)

        # No solution available, add only one value for smart_cc_share
        if 'smart_cc_share' in inputs.optimization_thetas:
            x0 = x0 if x0 is not None else \
                inputs.initial_theta.drop('smart_cc_share', level=1, axis=1).dropna(axis=1).iloc[0].tolist() + [
                    inputs.initial_theta.loc[:, pd.IndexSlice[:, 'smart_cc_share']].mean().mean()]
        else:
            x0 = x0 if x0 is not None else inputs.initial_theta.dropna(axis=1).iloc[0].tolist()

        if inputs.optimize:
            if 'smart_cc_share' in inputs.optimization_thetas:
                bounds = inputs.bounds.drop('smart_cc_share', axis=0)
                bound_lower = bounds.lower.tolist() * 4 if inputs.daytime_dependency else bounds.lower.tolist()
                bound_lower.append(inputs.bounds.lower.loc['smart_cc_share'])
                bound_upper = bounds.upper.tolist() * 4 if inputs.daytime_dependency else bounds.upper.tolist()
                bound_upper.append(inputs.bounds.upper.loc['smart_cc_share'])
            else:  # 'smart_cc_share' not in optimization thetas
                bounds = inputs.bounds
                bound_lower = bounds.lower.tolist() * 4 if inputs.daytime_dependency else bounds.lower.tolist()
                bound_upper = bounds.upper.tolist() * 4 if inputs.daytime_dependency else bounds.upper.tolist()

            # Make sure that popsize is at least cfg.popsize and a multiple of number_of_processes to be efficient
            popsize = min(cfg_popsize, number_of_processes)
            while popsize < cfg_popsize:
                popsize += number_of_processes

            options = {
                'bounds': [bound_lower, bound_upper],
                'maxfevals': max_episodes,
                'popsize': popsize}

            es = cma.CMAEvolutionStrategy(
                x0 if x0 is not [] else [0, 0],  # [] if no thetas have been selected -> replace with [0, 0]
                cfg.std_deviation,
                options)
            if best_cost is not None:
                es.best.f = best_cost  # Set best available run as best solution

    return es, max_episodes


def optimization(inputs, number_of_processes=cfg.popsize, max_episodes=cfg.max_episodes, cfg_popsize=cfg.popsize,
                 fixed=None, new=False, new_avrg_cc=False):
    from cma.fitness_transformations import EvalParallel2

    logger.warning('\n==================================================================================='
                   '\nStarting optimization with the following PEV file: ...{}'
                   '\n==================================================================================='
                   ''.format(inputs.pev_file[-50:]))
    logger.warning('MainModel running on process {}'.format(os.getpid()))

    # Saving input parameters
    with open('{}/Outputs/{}_parameters.dill'.format(cfg.parent_folder, inputs.name), 'wb') as f:
        dill.dump(inputs, f)

    logger.info('=== STARTING CMA ALGORITHM ===')
    inputs.parallel_processing = False
    cma_file = '{}/Outputs/{}_CMA.dill'.format(cfg.parent_folder, inputs.name)

    # Parallel evaluations: https://github.com/CMA-ES/pycma/issues/31
    with EvalParallel2(fitness_function=fitness,
                       number_of_processes=number_of_processes,
                       maxtasksperchild=1) as eval_all:
        es, max_episodes = get_es(inputs, cma_file, fixed, cfg_popsize, number_of_processes, max_episodes)
        while not es.stop() and es.countevals <= max_episodes:
            x = es.ask()
            es.tell(x, eval_all(x, args=(inputs,
                                         fixed,
                                         copy.deepcopy(es.countevals),
                                         copy.deepcopy(es.best.f),
                                         False,
                                         new,
                                         new_avrg_cc)))
            logger.warning('\n==================================================================================='
                           '\nCurrent status of CMAES: Evaluations = {}, Best solution = {} at evaluation {}'
                           '\n==================================================================================='
                           ''.format(es.result.evaluations, es.result.fbest, es.result.evals_best))
            with open(cma_file, 'wb') as f:
                dill.dump(es, f)
            cfg.aws_upload_results(inputs.name, temp=True)
        # eval_all.terminate()

    logger.info('=== Lowest overall cost: {} === \n Thetas for lowest cost: {}'.format(es.best.f, es.best.x))
    cfg.aws_upload_results(inputs.name)


def testing_calc(inputs, max_parallel, only_g2v=False, only_ucc=False, test_number=0, charging_tech='G2V',
                 one_more_iteration=True, invest_not_total=False, original_grid=False, fixed=None, new=False,
                 number_of_sets=None, number_of_tests=cfg.number_of_tests, combined_expansion=None,
                 remove_scenario_files=cfg.remove_scenario_files, remove_outputs_test=cfg.remove_outputs_test,
                 new_avrg_cc=False, new_rides=False):
    import copy
    import multiprocessing as mp
    from functools import partial
    from glob import glob
    from Functions import \
        pev_functions as pev, \
        grid_functions as grd

    def get_test_dict(episode):
        # Get inputs
        inputs_dict = inputs.get_inputs_dict(charging_tech=charging_tech,
                                             episode=episode,
                                             number_of_sets=number_of_sets)  # inputs_dict = {0: inputs_dict[0]}

        # Get PEV scenarios
        pev.get_set_of_pev(inputs_dict,
                           new_rides=True,
                           max_parallel=max_parallel)
        # Get load scenarios
        grd.get_set_of_loads(inputs_dict,
                             max_parallel=max_parallel)

        # Update number of chargers to max across all scenarios
        pev.update_number_chargers(inputs_dict, number_of_sets=number_of_sets, testing=True)

        inputs_dict_opttest = copy.deepcopy(inputs_dict)
        inputs_dict_ucctest = copy.deepcopy(inputs_dict)
        for rides_set in inputs_dict_ucctest.values():
            rides_set.charging_tech = 'UCC'
            rides_set.curtailment = False

        if not only_g2v and not only_ucc:
            return {'test_best_g2vucc': inputs_dict_opttest,
                    'test_ucc': inputs_dict_ucctest}
        elif only_g2v:
            return {'test_best_g2vucc': inputs_dict_opttest}
        elif only_ucc:
            return {'test_ucc': inputs_dict_ucctest}

    # Number of sets that are evaluated at the same time
    annuities = 99999
    if number_of_sets is None:
        number_of_sets = min(10, number_of_tests)
    if combined_expansion is None:
        combined_expansion = True if number_of_sets > 1 else False

    while one_more_iteration:
        logger.warning('\n==================================================================================='
                       '\nStarting test run {} for UCC and G2VUCC with current annuities of {}'
                       '\n==================================================================================='
                       ''.format(test_number, annuities))

        # Set up the PEV sets
        inputs.update_worst_best_files(test_number=test_number,
                                       combined_expansion=combined_expansion,
                                       invest_not_total=invest_not_total,
                                       fixed=fixed)
        inputs.safety_margin = 1.0

        test_dict = get_test_dict(episode=9000 + (number_of_sets * test_number))
        test_combined_expansion_prep(test_dict,
                                     max_parallel,
                                     test_number,
                                     original_grid=original_grid,
                                     combined_expansion=combined_expansion,
                                     new=new,
                                     new_avrg_cc=new_avrg_cc)
        logger.info('Expansion preparation for iteration {} done'.format(test_number))

        if combined_expansion:
            pool = mp.Pool(processes=2,
                           maxtasksperchild=1)
            func = partial(test_combined_expansion_calc,
                           test_dict=test_dict,
                           test_number=test_number,
                           original_grid=original_grid)
            results = pool.map(func, test_dict.keys(),
                               chunksize=1)
            pool.close()
            pool.join()

            annuities_test_g2vucc, annuities_test_ucc = results[0], results[1]
            annuities = annuities_test_ucc + annuities_test_g2vucc
        else:
            cost = pd.read_csv('{}_cost_best_test.csv'.format(inputs.out_file), index_col=[0])
            annuities_test_ucc = cost.query('type == "test_ucc"').iloc[-number_of_sets:].sum(0).annuities_grid
            annuities_test_g2vucc = cost.query('type == "test_best_g2vucc"'
                                               ).iloc[-number_of_sets:].sum(0).annuities_grid

        one_more_iteration = not ((annuities == 0) and
                                  (test_number + 1 >= number_of_tests * 2 / number_of_sets))
        logger.info('Need to run another iteration? {}'.format(one_more_iteration))
        logger.warning('\n==================================================================================='
                       '\nTest iteration {} finished with EUR {} Annuities UCC & EUR {} Annuities G2VUCC'
                       '\n==================================================================================='
                       ''.format(test_number, annuities_test_ucc, annuities_test_g2vucc))
        cfg.aws_upload_results(inputs.name,
                               temp=True)

        if one_more_iteration and test_number > 0:
            # Remove scenario files
            if remove_scenario_files:
                for inputs in test_dict[list(test_dict.keys())[0]].values():
                    cfg.try_remove_file(inputs.grid_file_scenario)
                    cfg.try_remove_file(inputs.pev_file)

            # Remove output files for previous test run
            if remove_outputs_test and test_number > 1:
                for name in ['test_best_g2vucc', 'test_ucc']:
                    files_old = glob('{}_{}_t{}_p*_c*'.format(inputs.out_file, name, test_number - 1))
                    for file_old in files_old:
                        cfg.try_remove_file(file_old)

            # Remove all temp files that were used for the test runs
            if cfg.remove_test_timeseries:
                for name in ['test_best_g2vucc', 'test_ucc']:
                    for inputs_id, inputs in zip(list(test_dict[name].keys()),
                                                 list(test_dict[name].values())):
                        out_files = glob('{}_{}_e{}_c*.dill'.format(
                            inputs.out_file, name, inputs.episode))
                        for out_file in out_files:
                            cfg.try_remove_file(out_file)

        if one_more_iteration:
            test_number += 1

    # Calculate the grid losses for each of the scenarios individually
    logger.warning('\n==================================================================================='
                   '\nStarting calculation of grid losses for individual out-of-sample scenarios'
                   '\n==================================================================================='
                   ''.format(test_number, annuities))
    test_dict = get_test_dict(episode=9000 + (number_of_sets * test_number))
    inputs_names = []
    for name in test_dict.keys():
        for inputs in list(test_dict[name].values()):
            inputs_names.append([inputs, name])

    pool = mp.Pool(processes=max_parallel,
                   maxtasksperchild=1)
    func = partial(test_final_losses,
                   test_number=test_number,
                   combined_expansion=combined_expansion,
                   original_grid=False,
                   new=new,
                   new_avrg_cc=new_avrg_cc)
    pool.map(func,
             inputs_names,
             chunksize=1)
    pool.close()
    pool.join()

    cfg.aws_upload_results(inputs.name)
