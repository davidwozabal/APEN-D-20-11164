import os
import dill
import copy
import pandas as pd
import numpy as np
from Functions import config_functions as cfg
import logging
logger = logging.getLogger(__name__)

# from line_profiler import LineProfiler
# import profile
#
# def do_profile(follow=[]):
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


class DistributionGrid:
    __slots__ = ('year', 'edisgo', 'pd_load', 'qd_load', 'pp_load', 'qp_load', 'pcurtail', 'qcurtail', 'consumption',
                 'pkw_number', 'equipment',
                 'trafo_s_original', 'line_s_original', 'over_v_original', 'under_v_original', 'transformer_s',
                 'line_s', 'transformer_sallowed', 'line_sallowed',  'nodes', 'loops', 'line_selections',
                 'load_case', 'random_sample', 'timesteps', 'problems_post')

    def __init__(self, year):
        """
        eDisGo documentation: https://edisgo.readthedocs.io/en/dev/start_page.html
        eDisGo config file:
        https://github.com/openego/eDisGo/blob/dev/edisgo/config/config_grid_expansion_default.cfg#L69-L80
        eGo documentation: https://openego.readthedocs.io/en/dev/index.html
        """

        self.year = year
        self.consumption = pd.DataFrame([], index=['agricultural', 'industrial', 'residential', 'retail'])
        self.edisgo = None
        self.trafo_s_original = None
        self.line_s_original = None
        self.over_v_original = None
        self.under_v_original = None
        self.pkw_number = 0
        self.pd_load = np.array([], dtype='float32')  # MW
        self.qd_load = np.array([], dtype='float32')  # MW
        self.pp_load = np.array([], dtype='float32')  # MW
        self.qp_load = np.array([], dtype='float32')  # MW
        self.pcurtail = np.array([], dtype='float32')  # MW
        self.qcurtail = np.array([], dtype='float32')  # MW
        self.equipment = pd.DataFrame(columns=['name', 'type', 'product', 'length_kva', 'voltage'])
        self.nodes = {}
        self.loops = {}
        self.line_selections = None

        self.transformer_s = np.array([], dtype='float32')
        self.line_s = np.array([], dtype='float32')
        self.transformer_sallowed = np.array([], dtype='float32')
        self.line_sallowed = np.array([], dtype='float32')
        self.load_case = np.array([], dtype='float32')
        self.random_sample = None
        self.timesteps = {'initial': None,
                          'original': None,
                          'final': None}
        self.problems_post = {}

    def back_to_dataframe(self, inputs, loads=False, only_cluster_days=False):  # grid_parameters=False,
        def as_dataframe(data):
            if data.shape[0]/(365*cfg.t_periods) <= 1:  # Not combined expansion test data
                index = inputs.dates_represented_index if only_cluster_days else inputs.index
            else:
                index = pd.date_range('2011-01-01',
                                      periods=data.shape[0],
                                      freq=str(24 / cfg.t_periods) + 'H')
            try:
                return pd.DataFrame(data, index=index, columns=self.nodes['node'])
            except ValueError:
                try:
                    return pd.DataFrame(data[0, :, :], index=index, columns=self.nodes['node'])
                except IndexError:
                    return pd.DataFrame(0, index=index, columns=self.nodes['node'])

        if loads:
            self.pd_load = as_dataframe(self.pd_load)  # [kW]
            self.qd_load = as_dataframe(self.qd_load)  # [kW]
            self.pp_load = as_dataframe(self.pp_load)  # [kW]
            self.qp_load = as_dataframe(self.qp_load)  # [kW]
            self.pcurtail = as_dataframe(self.pcurtail)
            self.qcurtail = as_dataframe(self.qcurtail)

    def node_aggregation(self, load_input, load_type='demand'):
        """ Takes load curves for all nodes as input, where each column of DataFrame is either a MV node for a specific
        load type or a LV node. Returns load curves for all node, where each column is an aggregated MV or LV node"""
        import re

        if load_type == 'demand':
            load = copy.deepcopy(load_input)
            n_names = []
            for column in load.columns:  # Align names of demand loads to get base load without PEV
                if 'Load_aggregated' in column:
                    name = column
                else:
                    name = re.split('_', str(column))[-1]
                n_names.append(name)
            load.columns = n_names
            load = load.groupby(level=0, axis=1).sum()

        if load_type == 'supply':
            load = copy.deepcopy(load_input)
            subtypes = {}
            for n in self.edisgo.network.mv_grid.graph.nodes():
                try:
                    subtypes[repr(n)] = n.subtype
                except AttributeError:
                    pass

            n_names = []
            for column in load.columns:  # Align names of generation loads to get base load without PEV
                if column in subtypes.keys():
                    name = column
                else:
                    name = re.split('_', str(column))[-1]
                n_names.append(name)
            load.columns = n_names
            load = load.groupby(level=0, axis=1).sum()
            try:  # Drop slack bus if available
                load = load.drop('slack', axis=1)
            except ValueError:
                pass

        return load.round(cfg.rounding).astype('float32')

    def get_loads_from_oedb(self, inputs):
        """Set up the EDisGo object using the OpenEnergy DataBase and the oemof demandlib to set up time series for
        loads and fluctuating generators (time series for dispatchable generators need to be provided)"""
        import time
        import oedialect.engine as oe

        tries = cfg.oedb_acces_tries
        for trynumber in range(tries):
            try:
                try:
                    self.edisgo.import_generators(generator_scenario=cfg.scenario[inputs.year])  # Import generators
                    logger.info('Grid {} successfully imported for year {} from OEDB'.format(
                        inputs.edisgo_grid, inputs.year))
                    return None
                except ValueError as e:
                    return e
            except oe.ConnectionException as e:
                logger.info('Grid {} generator import for year {} from OEDB FAILED, try {}/{}, waiting {} seconds\n'
                            'ERROR: {}'.format(inputs.edisgo_grid, inputs.year, trynumber, tries, cfg.waittime, e))
                time.sleep(cfg.waittime)
                if trynumber + 1 == tries:
                    raise AssertionError('Error in getting loads from oedb: {}: {}'.format(type(e), e))

    @staticmethod
    def get_timeseries(edisgo, timeindex, inputs):
        """Get a timeseries for the edisgo object and retry in case of problems with the oedb database"""
        from edisgo.grid.network import TimeSeriesControl
        import time

        tries = cfg.oedb_acces_tries
        for trynumber in range(tries):
            try:
                return TimeSeriesControl(
                    network=edisgo.network,
                    timeseries_generation_fluctuating='oedb',
                    timeseries_generation_dispatchable=pd.DataFrame({'other': [1] * len(timeindex)}, index=timeindex),
                    timeseries_generation_reactive_power=None,
                    timeseries_load='demandlib',
                    timeseries_load_reactive_power=None,
                    timeindex=timeindex).timeseries
            except Exception as e:
                wait = np.random.randint(cfg.waittime)
                logger.info('Timeseries import for grid {} from OEDB FAILED, try {}/{}, waiting {} seconds\n'
                            'ERROR: {}'.format(inputs.edisgo_grid, trynumber, tries, wait, e))
                time.sleep(wait)
                if trynumber + 1 == tries:
                    raise AssertionError('Error in get_timeseries() after {} tries: {}'.format(tries, e))

    def interpolate_loads(self, inputs, full_year):
        """ Extracts demand and generation loads from the edisgo object for the given grid. Interpolation is performed
        if t_periods > 24
            - edisgo.network.pypsa.loads_t.keys(): 'p_set', 'q_set', 'p', 'q'
            - edisgo.network.pypsa.generators_t.keys(): 'p_min_pu', 'p_max_pu', 'p_set', 'q_set', 'p', 'q', 'status' """

        def interpolate_loads(load):
            return self.edisgo.network.pypsa.loads_t[load].reindex(
                inputs.index if full_year else inputs.dates_represented_index
                ).interpolate(method='cubic').ffill().clip(lower=0)

        def interpolate_generators(load):
            return self.edisgo.network.pypsa.generators_t[load].reindex(
                inputs.index if full_year else inputs.dates_represented_index
                ).interpolate(method='cubic').ffill().clip(lower=0)

        # Interpolate load curves from edisgo_temp and set to self.edisgo
        self.edisgo.network.timeseries.load = self.edisgo.network.timeseries.load.reindex(
            inputs.index).interpolate(method='cubic').ffill().clip(lower=0)
        self.edisgo.network.timeseries.generation_fluctuating = \
            self.edisgo.network.timeseries.generation_fluctuating.reindex(
                inputs.index).interpolate(method='cubic').ffill().clip(lower=0)

        self.edisgo.network.pypsa.loads_t.p = interpolate_loads('p')
        self.edisgo.network.pypsa.loads_t.p_set = interpolate_loads('p_set')
        self.edisgo.network.pypsa.loads_t.q = interpolate_loads('q')
        self.edisgo.network.pypsa.loads_t.q_set = interpolate_loads('q_set')

        self.edisgo.network.pypsa.generators_t.p = interpolate_generators('p')
        self.edisgo.network.pypsa.generators_t.p_set = interpolate_generators('p_set')
        self.edisgo.network.pypsa.generators_t.q = interpolate_generators('q')
        self.edisgo.network.pypsa.generators_t.q_set = interpolate_generators('q_set')

    def update_loads_and_consumption(self, inputs):
        """Converts all pypsa load and generation timeseries to numpy array for each MV node in the grid"""
        import re

        # Get loads from edisgo object, all [MW], p_set because Generator_slack not included
        pd_load = self.node_aggregation(self.edisgo.network.pypsa.loads_t.p, load_type='demand')  # [MW]
        qd_load = self.node_aggregation(self.edisgo.network.pypsa.loads_t.q, load_type='demand')  # [MW]
        pp_load = self.node_aggregation(self.edisgo.network.pypsa.generators_t.p, load_type='supply')  # [MW]
        qp_load = self.node_aggregation(self.edisgo.network.pypsa.generators_t.q, load_type='supply')  # [MW]

        # Save nodes
        list_node = list(set(list(pd_load.columns) + list(pp_load.columns)))
        list_node.sort()
        self.nodes['n_id'] = list(range(len(list_node)))
        self.nodes['node'] = list_node
        self.nodes['n_full_load'] = []  # np.empty(len(self.nodes['node']))
        self.nodes['n_full_res'] = []  # np.empty(len(self.nodes['node']))

        # Make sure all nodes are everywhere
        all_nodes = pd.DataFrame(0, index=inputs.index, columns=self.nodes['node'])
        self.pd_load = copy.deepcopy(all_nodes).add(pd_load, fill_value=0).loc[:, self.nodes['node']].values
        self.qd_load = copy.deepcopy(all_nodes).add(qd_load, fill_value=0).loc[:, self.nodes['node']].values
        self.pp_load = copy.deepcopy(all_nodes).add(pp_load, fill_value=0).loc[:, self.nodes['node']].values
        self.qp_load = copy.deepcopy(all_nodes).add(qp_load, fill_value=0).loc[:, self.nodes['node']].values

        # Get the consumption per year
        self.consumption = pd.DataFrame([], index=['agricultural', 'industrial', 'residential', 'retail'])
        for mv_node in self.edisgo.network.mv_grid.loads:  # Generate columns for each MV load cluster
            load_type = re.split('_', repr(mv_node))[1]  # After the first '_'
            self.consumption.loc[load_type, repr(mv_node)] = mv_node.consumption[load_type]
        for lv_grid in self.edisgo.network.mv_grid.lv_grids:  # All LV grids in edisgo
            n_name = re.split('_', repr(lv_grid))[-1]
            self.consumption[n_name] = lv_grid.consumption
        self.consumption.fillna(0, inplace=True)

    def setup_edisgo_grid(self, inputs, file_gridobjecttemp, get_res=True, full_year=False, save_temp=True,
                          scenario_only=False):
        from edisgo.tools import pypsa_io
        from Functions import loadflow_functions as lff

        logger.info('Setting up the eDisGo grid number {} for year {} on process {}'.format(
            inputs.edisgo_grid, inputs.year, os.getpid()))
        error = ''

        file_edisgoinitial = '{}/Grids/Grid_[{}]_edisgoinitial_t{}_y{}.pkl'.format(
            cfg.parent_folder, inputs.edisgo_grid, inputs.index.shape[0], inputs.year)
        file_edisgoscenario = '{}/Grids/Grid_[{}]_edisgoscenario_t{}_y{}_RES{}.dill'.format(
            cfg.parent_folder, inputs.edisgo_grid, inputs.index.shape[0], inputs.year, get_res)

        if os.path.isfile(file_gridobjecttemp):
            with open(file_gridobjecttemp, 'rb') as f:
                self.edisgo = dill.load(f)
        else:
            if os.path.isfile(file_edisgoscenario):
                with open(file_edisgoscenario, 'rb') as f:
                    self.edisgo = dill.load(f)
            else:
                with open('{}/Inputs/grid_pkw_numbers.pkl'.format(cfg.parent_folder), 'rb') as f:
                    pkw_numbers = dill.load(f)
                self.pkw_number = pkw_numbers[pkw_numbers.grid_id == inputs.edisgo_grid].PKW_gridPLZ.sum()

                # Initial edisgo grid object with initial reinforcement
                self.edisgo = get_initial_edisgo_grid(inputs,
                                                      file_edisgoinitial,
                                                      save=cfg.save_grid_files)
                self.edisgo.network.timeseries = \
                    self.get_timeseries(self.edisgo,
                                        inputs.index if full_year else inputs.dates_represented_index,
                                        inputs)  # Replace Timeindex
                if get_res:
                    error = self.get_loads_from_oedb(inputs)  # Get edisgo object

                # All info downloaded from oedb
                with open(file_edisgoscenario, 'wb') as f:
                    dill.dump(self.edisgo, f)
                try:
                    os.remove(file_edisgoinitial)
                    os.remove(inputs.grid_file)
                except FileNotFoundError:
                    pass

                if scenario_only:
                    return

            logger.info('Getting pypsa object for grid {} and year {} on process {}'.format(
                inputs.edisgo_grid, inputs.year, os.getpid()))
            self.edisgo.network.pypsa = \
                pypsa_io.to_pypsa(self.edisgo.network,
                                  mode='mv',
                                  timesteps=inputs.index if full_year else inputs.dates_represented_index)

            # Loads from edisgo_temp to self.edisgo, interpolated timeseries
            self.interpolate_loads(inputs, full_year)
            timesteps = inputs.index if full_year else inputs.dates_represented_index
            logger.info('Analyzing grid {} for year {}'.format(inputs.edisgo_grid, inputs.year))
            self.edisgo, timesteps = lff.analyze(self.edisgo,
                                                 mode='mv',
                                                 timesteps=timesteps)
            if save_temp:
                with open(file_gridobjecttemp, 'wb') as f:
                    dill.dump(self.edisgo, f)
                try:
                    os.remove(file_edisgoinitial)
                except FileNotFoundError:
                    pass

        self.update_loads_and_consumption(inputs)

        return error

    def get_equipment(self):
        equipment = pd.DataFrame(columns=['name', 'type', 'product', 'length_kva', 'voltage'])

        item_num = 0
        # Get lines from the lv-level
        for g in self.edisgo.network.mv_grid.lv_grids:
            for e in g.graph.edge.keys():
                edge = g.graph.edge[e]
                for item in edge.keys():
                    equipment.loc[item_num, :] = [edge[item]['line'], edge[item]['type'],
                                                  edge[item]['line'].type.name, edge[item]['line'].length, 'lv']
                    item_num += 1

        # Get lines from the mv-level
        for e in self.edisgo.network.mv_grid.graph.edge.keys():
            edge = self.edisgo.network.mv_grid.graph.edge[e]
            for item in edge.keys():
                equipment.loc[item_num, :] = [edge[item]['line'], edge[item]['type'],
                                              edge[item]['line'].type.name, edge[item]['line'].length, 'mv']
                item_num += 1

        # Get transformers
        for node in self.edisgo.network.mv_grid.graph.nodes():
            try:  # Skip nodes that do not contain transformers
                transformers = node.transformers
                for item in transformers:
                    equipment.loc[item_num, :] = [item.id, 'transformer', item.type.S_nom, item.type.S_nom, 'mv']
                    item_num += 1
            except AttributeError:
                pass

        self.equipment = equipment.drop_duplicates().reset_index().loc[:, ['voltage', 'type', 'length_kva']]

    def get_equipment_expansion(self):
        if self.equipment.index.empty:
            self.get_equipment()

        # Calculate share of original line km and transformer kVA
        length_mv_lines = self.equipment[(self.equipment.voltage == 'mv') &
                                         (self.equipment.type == 'line')].length_kva.sum()
        kva_transformers = self.equipment[(self.equipment.voltage == 'mv') &
                                          (self.equipment.type == 'transformer')].length_kva.sum()

        # Get new line km and transformer kVA
        kva_new_transformers, length_new_mv_lines = 0, 0
        try:
            for item in self.edisgo.network.results.grid_expansion_costs.index:
                try:  # Try if item is a transformer, if yes add to new transformers
                    kva_new_transformers += item.type.S_nom
                except AttributeError:  # Item is a line and not a transformer, add to new lines
                    try:
                        length_new_mv_lines += item.length
                    except AttributeError:
                        logger.error(item)
        except AttributeError:  # No equipment could be expanded
            pass

        return [round(length_new_mv_lines / length_mv_lines, cfg.rounding),
                round(kva_new_transformers / kva_transformers, cfg.rounding)]

    def get_aggregated_overloads(self, inputs, transformer=False, line=False, voltage=False, return_allowed=False,
                                 return_is=False, original=False, incl_hvmv=True, network=None, timesteps=None,
                                 lf_results=False, level='mv'):
        """ inputs object only needed if original == True"""
        import re
        from Functions import loadflow_functions as lff

        def full_np(load_df):
            """ Convert to np.array that includes all nodes, especially adds "ghost" nodes for RES directly connected
             to the MV-Level """

            if return_allowed and transformer:
                fill_value = 999999
            elif voltage:
                fill_value = 1
            else:
                fill_value = 0

            load_return = pd.DataFrame(0,
                                       index=load_df.index,
                                       columns=self.nodes['node']
                                       ).add(load_df.loc[:, self.nodes['node']],
                                             fill_value=fill_value)

            return load_return.values.astype('float32')

        if network is None:
            if original:
                grid = get_grid(inputs)
                network = grid.edisgo.network
                timesteps = grid.timesteps['final']
            else:
                network = self.edisgo.network
                timesteps = self.timesteps['initial']

        load = pd.DataFrame()
        if transformer:
            if incl_hvmv:  # Including HV/MV transformer
                load = lff.get_transformer_overloads(network,
                                                     network.mv_grid.station,
                                                     grid_level='mv',
                                                     return_allowed=return_allowed,
                                                     return_is=return_is)  # HV/MV
            for lv_grid in network.mv_grid.lv_grids:  # MV/LV
                load = pd.concat([load, lff.get_transformer_overloads(network,
                                                                      lv_grid.station,
                                                                      grid_level='lv',
                                                                      return_allowed=return_allowed,
                                                                      return_is=return_is)], axis=1)
            load.columns = [re.split('_', str(node))[-1] for node in load.columns]
            load = full_np(load)  # Convert to np.array that includes all nodes

        if line:
            if lf_results or return_is:
                load = lff.get_line_overloads(network,
                                              return_allowed=return_allowed,
                                              return_is=return_is)
            elif return_allowed:
                load = pd.Series()  # s_line_allowed
                for line in list(network.mv_grid.graph.lines()):
                    if inputs.edisgo_grid == 163 or inputs.edisgo_grid == 2948:  # Test
                        s_max = (line['line'].type['I_max_th'] * line['line'].type['U_n'] / 3
                                 ) * line['line'].quantity  # s_max = u_nom * i_max / 3
                    else:
                        s_max = (line['line'].type['I_max_th'] * line['line'].type['U_n'] / (3 ** 0.5)
                                 ) * line['line'].quantity  # s_max = u_nom * i_max / (3 ** 0.5)
                    load[repr(line['line'])] = s_max

        if voltage:
            if return_allowed:
                if level == 'lv':
                    load = lff.get_voltage_allowed_lv(network)
                    load = [full_np(load[0]), full_np(load[1])]
                elif level == 'mv':
                    load = lff.get_voltage_allowed_mv(network)
                    load = [load[0].values, load[1].values]
                else:
                    logger.error('Level not correctly defined')
            else:
                load = full_np(lff.get_voltage_is(network, level=level))  # Convert to np.array that includes all nodes

        return load

    def change_demand_loads(self, pevs, inputs, test_number=None):
        """ Takes the pev charging loads from the PEVInfo object and translates them into DataFrames with nodes as
        columns and index as time steps
        - ppev_load includes only columns for each node in total just like pp_load, pd_load, etc. """
        import re

        charging = pd.DataFrame(pevs.ppev_load,
                                index=inputs.index,
                                columns=self.nodes['node']).div(1e3)  # [kW] -> [MW]
        # charging = charging.loc[:, (charging != 0).any(axis=0)]  # Remove nodes without PEV charging

        charging_p_set = pd.DataFrame(0, index=inputs.index, columns=self.edisgo.network.pypsa.loads_t.p_set.columns)
        for n in charging_p_set.columns:
            if 'Load_aggregated' in n or 'GeneratorFluctuating' in n:
                name = n
            else:
                name = re.split('_', n)[-1]
            charging_p_set.loc[:, n] = charging.loc[:, name]

        # Add safety margin
        if (test_number is None) or (test_number == 0):
            charging_p_set *= inputs.safety_margin

        # Add charging load to grid data, including charging and discharging inefficiencies [MW]
        self.edisgo.network.pypsa.loads_t.p_set = self.edisgo.network.pypsa.loads_t.p_set.add(
                                                  charging_p_set, fill_value=0)
        return pevs

    def downsample_grid(self, inputs):
        logger.info('Downsampling grid {} to {}'.format(inputs.edisgo_grid, inputs.downsample))

        if inputs.downsample:
            self.edisgo.network.pypsa.generators_t.p_set = \
                self.edisgo.network.pypsa.generators_t.p_set.resample(inputs.downsample).max()
            self.edisgo.network.pypsa.generators_t.q_set = \
                self.edisgo.network.pypsa.generators_t.q_set.resample(inputs.downsample).max()
            self.edisgo.network.pypsa.generators_t.p = \
                self.edisgo.network.pypsa.generators_t.p.resample(inputs.downsample).max()
            self.edisgo.network.pypsa.generators_t.q = \
                self.edisgo.network.pypsa.generators_t.q.resample(inputs.downsample).max()
            self.edisgo.network.pypsa.loads_t.p_set = \
                self.edisgo.network.pypsa.loads_t.p_set.resample(inputs.downsample).max()
            self.edisgo.network.pypsa.loads_t.q_set = \
                self.edisgo.network.pypsa.loads_t.q_set.resample(inputs.downsample).max()
            self.edisgo.network.pypsa.loads_t.p = self.edisgo.network.pypsa.loads_t.p.resample(inputs.downsample).max()
            self.edisgo.network.pypsa.loads_t.q = self.edisgo.network.pypsa.loads_t.q.resample(inputs.downsample).max()

    def update_load_case(self, t=None, pev_load=None, return_s=False, get_currents_s=False):
        pd_load = self.pd_load[t] if t is not None else self.pd_load
        qd_load = self.qd_load[t] if t is not None else self.qd_load
        pp_load = self.pp_load[t] if t is not None else self.pp_load
        qp_load = self.qp_load[t] if t is not None else self.qp_load

        # Demand load plus PEV charging load at node at time t (Dtn(St) = Dctn + Dvtn)
        sd_load = calc_apparent_power(pd_load if pev_load is None else pd_load + pev_load / 1E3,  # [kW] -> [MW]
                                      qd_load)  # App.Power [kVA]
        # Generation load (supply) at node at time t (Srtn)
        sp_load = calc_apparent_power(pp_load,
                                      qp_load)  # Apparent Power per MV nodes [kVA]
        transformer_s = (sd_load - sp_load) * 1E3  # [MW] -> [kW]
        load_case = transformer_s > 0  # self.feedin_case = (sd_load - sp_load) <= 0

        if return_s:
            self.load_case = load_case

            return sd_load, sp_load

        if get_currents_s:
            # Get line_s from the loads of all relevant transformers
            line_s = self.nodes_to_lines(transformer_s)
            # Check if lines_s is correct with: line_s / self.edisgo.network.results.s_res().iloc[0,:49].values

            return transformer_s, line_s

    def add_curtailment(self, inputs):
        """ Takes the calculated curtailment amount from the and translates them into DataFrames with generation nodes
        as columns and index as time steps"""
        import re

        # Bring self.pcurtail into typical DataFrame
        curtail_p = pd.DataFrame(0,
                                 index=inputs.index,
                                 columns=self.edisgo.network.pypsa.generators_t.p_set.columns)

        for node in self.edisgo.network.pypsa.generators_t.p_set.columns:
            if 'Load_aggregated' in node or 'GeneratorFluctuating' in node:
                name = node
            else:
                name = re.split('_', node)[-1]
            n_id = np.where(np.array(self.nodes['node']) == name)[0][0]

            gen_types = pd.Series(index=[node])
            gen_type = self.edisgo.network.pypsa.generators.type.loc[node]
            gen_types.loc[node] = re.split('_', gen_type)[0] if gen_type is not '' \
                else re.split('_', node)[0]
            gen_types.sort_values(inplace=True)  # Sort by generation type

            curtailment_node = copy.deepcopy(self.pcurtail[:, n_id])
            # Iterate through generators at node and perform curtailment with the most expensive (wind) first
            for n_full in gen_types.index[::-1]:  # Start from the bottom with wind
                curtail_unit = np.minimum(curtailment_node,
                                          self.edisgo.network.pypsa.generators_t.p.loc[:, n_full])
                curtailment_node -= curtail_unit
                curtail_p.loc[:, n_full] += curtail_unit  # [MW]

        q_p_factor = self.edisgo.network.pypsa.generators_t.q_set.divide(self.edisgo.network.pypsa.generators_t.p_set)
        q_p_factor = q_p_factor.abs().fillna(0).replace(np.inf, 0)
        self.edisgo.network.pypsa.generators_t.p_set = self.edisgo.network.pypsa.generators_t.p_set.sub(
                                                       curtail_p)  # [MW]
        self.edisgo.network.pypsa.generators_t.q_set = self.edisgo.network.pypsa.generators_t.q_set.add(
                                                       curtail_p.mul(q_p_factor))  # [MW]
        q_p_factor = np.absolute(self.qp_load / self.pp_load)
        q_p_factor.fill(0)
        self.qcurtail = np.multiply(self.pcurtail, q_p_factor)

        return curtail_p

    def setup_plot(self, pevs, inputs, original=False):
        import matplotlib.pyplot as plt

        if not original:
            try:
                pevs.all_to_dataframe(inputs, self.nodes['node'])
            except AttributeError:  # If Basecost without PEV or PEV from combined expansion
                pass
        self.back_to_dataframe(inputs, loads=True)  # grid_parameters=True,

        sd_load = ((self.pd_load.pow(2) +
                    self.qd_load.pow(2)).pow(0.5))  # Demand
        sp_load = ((self.pp_load.pow(2) +
                    self.qp_load.pow(2)) ** 0.5).multiply(-1)  # Generation

        if not original:
            sp_load_curtail = sp_load.add(((self.pcurtail.pow(2) +
                                            self.qcurtail.pow(2)).pow(0.5)))  # Generation incl. curtail
            sdp_load_res = sd_load.add(sp_load, fill_value=0)  # Demand, generation incl. curtail
            if pevs is not None:
                try:  # If PEV from combined expansion
                    spev_load = pd.DataFrame(pevs.ppev_load / 1000,
                                             index=sp_load.index, columns=sp_load.columns)  # PEV load [kW] -> [MW]
                    spev_load_must = pd.DataFrame(pevs.ppev_load_must / 1000,
                                                  index=sp_load.index, columns=sp_load.columns)  # PEV load [kW] -> [MW]
                except (TypeError, AttributeError):
                    spev_load = pevs.ppev_load.div(1000)  # PEV load [kW] -> [MW]
                    spev_load_must = pevs.ppev_load_must.div(1000)  # PEV load [kW] -> [MW]
            else:  # If Basecost without PEV
                spev_load = pd.DataFrame(0, index=sp_load.index, columns=sp_load.columns)
                spev_load_must = pd.DataFrame(0, index=sp_load.index, columns=sp_load.columns)
            st_load = sd_load.add(spev_load, fill_value=0).add(sp_load_curtail, fill_value=0) if inputs.curtailment \
                else sd_load.add(spev_load, fill_value=0).add(sp_load, fill_value=0)  # Total load [MW]
        else:
            sp_load_curtail = pd.DataFrame(0, index=sd_load.index, columns=sd_load.columns)
            sdp_load_res = pd.DataFrame(0, index=sd_load.index, columns=sd_load.columns)
            spev_load = pd.DataFrame(0, index=sd_load.index, columns=sd_load.columns)
            spev_load_must = pd.DataFrame(0, index=sd_load.index, columns=sd_load.columns)
            st_load = sd_load.add(sp_load, fill_value=0)  # Total load [MW]

        plt.clf()
        cfg.set_design(palette_size=11)

        loads = {'sp_load': sp_load, 'sp_load_curtail': sp_load_curtail, 'sdp_load_res': sdp_load_res,
                 'st_load': st_load, 'sd_load': sd_load, 'spev_load': spev_load, 'spev_load_must': spev_load_must}

        if original:
            return loads
        else:
            return loads, pevs

    @staticmethod
    def setup_node_plot(node, sp_load, sp_load_curtail, rides_at_node):
        try:
            res_pp = sp_load[node]
        except KeyError:
            res_pp = pd.Series(0, index=sp_load.index)

        try:
            pev_connected = rides_at_node.loc[:, node]
        except KeyError:
            pev_connected = pd.Series(0, index=sp_load.index)

        try:
            sp_load_curtail_node = sp_load_curtail[node]
        except KeyError:
            sp_load_curtail_node = pd.Series(0, index=sp_load.index)

        return res_pp, sp_load_curtail_node, pev_connected

    @staticmethod
    def add_legend(plt):
        import seaborn as sns
        from matplotlib.lines import Line2D

        labels = ['Demand load', 'PEV load', 'PEV load "must"', 'RES load curtailed', 'RES load',
                  'Total /w PEV & Curt.', 'Total /wo PEV & Curt.', 'Remaining load delta', 'Original load delta',
                  'Available charging', 'Available discharging']
        styles = ['-', '-', ':', '-', ':', '-', ':', '-', ':', '--', '--']
        colors = [sns.color_palette()[0], sns.color_palette()[1], sns.color_palette()[1], sns.color_palette()[2],
                  sns.color_palette()[2], sns.color_palette()[3], sns.color_palette()[3], sns.color_palette()[7],
                  sns.color_palette()[7], sns.color_palette()[5], sns.color_palette()[6]]
        lines = [Line2D([0], [0], color=colors[i], linestyle=styles[i]) for i in range(len(labels))]
        plt.legend(lines, labels, loc='lower right', ncol=2, prop={'size': 10})

        return plt

    @staticmethod
    def two_scales(ax1, demand, pev, pev_must, res, res_u, demandres, total, original_load_delta,
                   remaining_load_delta, ccapa, dcapa, connected, width=0.02, limit_factor=1.5):
        import math
        import seaborn as sns

        ax2 = ax1.twinx()
        ax1.plot(demand, color=sns.color_palette()[0])
        ax1.plot(pev, color=sns.color_palette()[1])
        ax1.plot(pev_must, color=sns.color_palette()[1], linestyle=':')
        ax1.plot(res_u, color=sns.color_palette()[2])
        ax1.plot(res, color=sns.color_palette()[2], linestyle=':')
        ax1.plot(total, color=sns.color_palette()[3])
        ax1.plot(demandres, color=sns.color_palette()[3], linestyle=':')
        ax1.plot(remaining_load_delta / 1000, color=sns.color_palette()[7])  # [kW] -> [MW]
        ax1.plot(original_load_delta / 1000, color=sns.color_palette()[7], linestyle=':')  # [kW] -> [MW]

        if not math.isnan(ccapa.max()):
            rects = ax2.bar(connected.index, connected, width=width, alpha=0.075, color=sns.color_palette()[8])
            ax1.plot(ccapa / 1000, color=sns.color_palette()[5], linestyle='--')
            ax1.plot(dcapa / 1000, color=sns.color_palette()[6], linestyle='--')

            for rect in rects:  # Attach a text label above each bar in *rects*, displaying its height
                height = 0 if math.isnan(rect.get_height()) else rect.get_height()
                cars = '' if height == 0 else str(int(height))
                ax2.annotate(cars,
                             xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points", ha='center', va='bottom', alpha=.5)
        ax2.grid(None)
        ax2.set_ylabel('PEV connected at the node [absolute numbers]')
        abs_max = max([demand.abs().max(), pev.abs().max(), pev_must.abs().max(), res.abs().max(),
                       res_u.abs().max(), demandres.abs().max(), total.abs().max()])
        ax1.set_ylim([abs_max * -limit_factor, abs_max * limit_factor])
        ax1.set_ylabel('Load [MVA]')

    def plot_full_grid(self, file, pevs, day, loads, end_node=12, save_plot=True, plotting=True):
        import math
        import matplotlib.pyplot as plt

        rides_at_node = pevs.rides_at_node.loc[day]
        end_node = min(end_node, len(loads['st_load'].index))

        rows = math.floor(end_node ** .5)
        columns = math.ceil(end_node / rows)
        fig, ax_array = plt.subplots(rows, columns)

        n_plot = 0
        for i, ax_row in enumerate(ax_array):
            for j, ax_plot in enumerate(ax_row):
                ax_plot.set_xticklabels([])
                if i == (rows - 1) and j == (columns - 1):  # Place sum values in last plot
                    self.two_scales(ax_plot,
                                    demand=loads['sd_load'].sum(axis=1),
                                    pev=loads['spev_load'].sum(axis=1),
                                    pev_must=loads['spev_load_must'].sum(axis=1),
                                    res=loads['sp_load'].sum(axis=1),
                                    res_u=loads['sp_load_curtail'].sum(axis=1),
                                    demandres=loads['sdp_load_res'].sum(axis=1),
                                    total=loads['st_load'].sum(axis=1),
                                    original_load_delta=pevs.original_load_delta[day].sum(axis=1),
                                    remaining_load_delta=pevs.remaining_load_delta[day].sum(axis=1),
                                    ccapa=pevs.fleet_charging_capa[day].sum(axis=1),
                                    dcapa=pevs.fleet_discharging_capa[day].sum(axis=1),
                                    connected=rides_at_node.sum(axis=1))
                    ax_plot.set_title('Total', loc='center', y=0.01)
                else:  # Plot each node into one plot
                    try:
                        node = self.pd_load.columns[n_plot]
                        res_pp, sp_load_curtail_node, pev_connected = \
                            self.setup_node_plot(node, loads['sp_load'], loads['sp_load_curtail'], rides_at_node)
                        self.two_scales(ax_plot,
                                        demand=loads['sd_load'][node],
                                        pev=loads['spev_load'][node],
                                        pev_must=loads['spev_load_must'][node],
                                        res=res_pp,
                                        res_u=sp_load_curtail_node,
                                        demandres=loads['sdp_load_res'][node],
                                        total=loads['st_load'][node],
                                        original_load_delta=pevs.original_load_delta[node][day],
                                        remaining_load_delta=pevs.remaining_load_delta[node][day],
                                        ccapa=pevs.fleet_charging_capa[node][day],
                                        dcapa=pevs.fleet_discharging_capa[node][day],
                                        connected=pev_connected)
                        ax_plot.set_title('Node {}'.format(node), loc='center', y=0.01)
                    except (KeyError, IndexError) as e:  # Leave plot empty if all nodes have already been plotted
                        logger.error('Error in plotting plot {}: {}'.format(n_plot, e))
                        pass
                n_plot += 1

        plt = self.add_legend(plt)
        fig.tight_layout(pad=1)
        if save_plot:
            save_file = '{}-{}.png'.format(file[:-5], day)
            plt.savefig(save_file, dpi=500)
        if plotting:
            plt.show()
        plt.close()
        logger.info('Full grid plot created for day {}.'.format(day))

    def plot_one_node(self, save_file, pevs, node, n_id, day, loads, grid_stats, save_plot=True, plotting=True,
                      original=False, summing=False):
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        timesteps = self.edisgo.network.pypsa.generators_t.p.index
        try:
            rides_at_node = pd.DataFrame() if (original and not summing) else pevs.rides_at_node
        except AttributeError:  # If pevs dictionary from combined expansion test
            try:
                rides_at_node = pd.DataFrame() if (original and not summing) else \
                    pd.DataFrame(pevs['rides_at_node'], columns=self.nodes['node'], index=self.pd_load.index)
            except AttributeError:  # If basecost without PEV
                rides_at_node = pd.DataFrame()
        try:
            pev_connected = pd.Series(rides_at_node[:, n_id], index=timesteps) \
                if not summing else rides_at_node.sum(1)
        except KeyError:
            pev_connected = pd.Series(0, index=loads['sp_load'].index)

        try:
            pev_connected = pev_connected.loc[day].reset_index(drop=True)
        except KeyError:  # If basecost without PEV
            pass

        if not original:
            try:
                original_load_delta = pevs.original_load_delta.loc[day, node].reset_index(drop=True)
                remaining_load_delta = pevs.remaining_load_delta.loc[day, node].reset_index(drop=True)
                ccapa = pevs.fleet_cc_capa.loc[day, node].reset_index(drop=True)
                dcapa = pevs.fleet_dc_capa.loc[day, node].reset_index(drop=True)
            except (AttributeError, TypeError):  # If pevs dictionary from combined expansion test
                try:
                    original_load_delta = pd.Series(pevs.original_load_delta[:, n_id],
                                                    index=timesteps).loc[day].reset_index(drop=True)
                    remaining_load_delta = pd.Series(pevs.remaining_load_delta[:, n_id],
                                                     index=timesteps).loc[day].reset_index(drop=True)
                    ccapa = pd.Series(pevs.fleet_cc_capa[:, n_id],
                                      index=timesteps).loc[day].reset_index(drop=True)
                    dcapa = pd.Series(pevs.fleet_dc_capa[:, n_id],
                                      index=timesteps).loc[day].reset_index(drop=True)
                except AttributeError:  # If basecost without PEV
                    original_load_delta = pd.Series(0, index=loads['sd_load'].index)
                    remaining_load_delta = pd.Series(0, index=loads['sd_load'].index)
                    ccapa = pd.Series(0, index=loads['sd_load'].index)
                    dcapa = pd.Series(0, index=loads['sd_load'].index)
        else:
            original_load_delta = pd.Series(0, index=loads['sd_load'].index)
            remaining_load_delta = pd.Series(0, index=loads['sd_load'].index)
            ccapa = pd.Series(0, index=loads['sd_load'].index)
            dcapa = pd.Series(0, index=loads['sd_load'].index)

        fig, ax_array = plt.subplots(2, 1)
        ax_array[0].set_xticklabels([])
        self.two_scales(ax_array[0],
                        demand=loads['sd_load'],
                        pev=loads['spev_load'],
                        pev_must=loads['spev_load_must'],
                        res=loads['sp_load'],
                        res_u=loads['sp_load_curtail'],
                        demandres=loads['sdp_load_res'],
                        total=loads['st_load'],
                        original_load_delta=original_load_delta,
                        remaining_load_delta=remaining_load_delta,
                        ccapa=ccapa,
                        dcapa=dcapa,
                        connected=pev_connected,
                        width=0.005,
                        limit_factor=1)
        ticks = list(range(0, len(day), cfg.t_periods))
        ax_array[0].set_title('Node {}'.format(node), loc='center', y=0.01)  # 9='upper center'
        ax_array[0].set_xticks(ticks)
        ax_array[0] = self.add_legend(ax_array[0])

        # Plot grid state parameters on second plot (line load, trafo load and voltage)
        ax1_2 = ax_array[1].twinx()

        # Plot voltage on left axis
        timesteps_mask = [x in day for x in self.edisgo.network.pypsa.loads_t.p.index]
        ax_array[1].plot((grid_stats['v_mvside_is']).interpolate().bfill(),
                         color=sns.color_palette()[2], alpha=.6)
        ax_array[1].plot((grid_stats['v_lvside_is']).interpolate().bfill(),
                         color=sns.color_palette()[3], alpha=.6)
        try:
            v_mvside_min = pd.Series(index=grid_stats['v_mvside_is'].index)
            v_mvside_min.loc[grid_stats['v_mvside_is'].dropna().index] = grid_stats['v_mvside_min'][timesteps_mask]
            v_mvside_max = pd.Series(index=grid_stats['v_mvside_is'].index)
            v_mvside_max.loc[grid_stats['v_mvside_is'].dropna().index] = grid_stats['v_mvside_max'][timesteps_mask]
            ax_array[1].fill_between(list(range(len(grid_stats['v_mvside_is']))),
                                     v_mvside_min.interpolate().bfill().values,
                                     v_mvside_max.interpolate().bfill().values,
                                     color=sns.color_palette()[2],
                                     alpha=.05)

            v_lvside_min = pd.Series(index=grid_stats['v_lvside_is'].index)
            v_lvside_min.loc[grid_stats['v_lvside_is'].dropna().index] = grid_stats['v_lvside_min'].dropna().iloc[:, 0]
            v_lvside_max = pd.Series(index=grid_stats['v_lvside_is'].index)
            v_lvside_max.loc[grid_stats['v_lvside_is'].dropna().index] = grid_stats['v_lvside_max'].dropna().iloc[:, 0]
            ax_array[1].fill_between(list(range(len(grid_stats['v_lvside_is']))),
                                     v_lvside_min.interpolate().bfill().values,
                                     v_lvside_max.interpolate().bfill().values,
                                     color=sns.color_palette()[3],
                                     alpha=.05)
        except ValueError:
            try:
                v_mvside_min = pd.Series(index=grid_stats['v_mvside_is'].index)
                v_mvside_min.loc[grid_stats['v_mvside_is'].index] = grid_stats['v_mvside_min'][timesteps_mask]
                v_mvside_max = pd.Series(index=grid_stats['v_mvside_is'].index)
                v_mvside_max.loc[grid_stats['v_mvside_is'].index] = grid_stats['v_mvside_max'][timesteps_mask]
                ax_array[1].fill_between(list(range(len(grid_stats['v_mvside_is']))),
                                         v_mvside_min.interpolate().bfill().values,
                                         v_mvside_max.interpolate().bfill().values,
                                         color=sns.color_palette()[2],
                                         alpha=.05)

                v_lvside_min = pd.Series(index=grid_stats['v_lvside_is'].index)
                v_lvside_min.loc[grid_stats['v_lvside_is'].dropna().index] = grid_stats['v_lvside_min'].dropna().iloc[:, 0]
                v_lvside_max = pd.Series(index=grid_stats['v_lvside_is'].index)
                v_lvside_max.loc[grid_stats['v_lvside_is'].dropna().index] = grid_stats['v_lvside_max'].dropna().iloc[:, 0]
                ax_array[1].fill_between(list(range(len(grid_stats['v_lvside_is']))),
                                         v_lvside_min.interpolate().bfill().values,
                                         v_lvside_max.interpolate().bfill().values,
                                         color=sns.color_palette()[3],
                                         alpha=.05)
            except ValueError as e:
                print('Error: {}'.format(e))

        try:
            mv_v_original = (pd.Series(self.over_v_original['is'][:, n_id], index=self.timesteps['final']
                                       ).loc[day].reset_index(drop=True)).interpolate().bfill()
            lv_v_original = (pd.Series(self.under_v_original['is'][:, n_id], index=self.timesteps['final']
                                       ).loc[day].reset_index(drop=True)).interpolate().bfill()
            ax_array[1].plot(mv_v_original,
                             color=sns.color_palette()[2], linestyle=':', alpha=.6)
            ax_array[1].plot(lv_v_original,
                             color=sns.color_palette()[3], linestyle=':', alpha=.6)
        except AttributeError:
            pass

        # Plot line and trafo load on first right axis
        max_steps = self.timesteps['initial'].shape[0]
        for l_is, l_allowed, lorg_is, lorg_allowed, color in [
                [grid_stats['line_s'], grid_stats['line_sallowed'], self.line_s_original,
                 grid_stats['line_sallowed_original'], 1],
                [grid_stats['transformer_s'], grid_stats['trafo_sallowed'], self.trafo_s_original,
                 grid_stats['trafo_sallowed_original'], 0]]:

            lorg_is_combined = np.array([])
            while lorg_is.shape[0] < max_steps:
                lorg_is_combined = np.append(lorg_is_combined, lorg_is)

            ax1_2.plot(l_is.div(l_allowed).interpolate().bfill(),
                       color=sns.color_palette()[color], alpha=.6)
            try:
                # Select relevant lines
                if color == 1:
                    lines, selections = self.line_selections
                    lorg_is = lorg_is.loc[:, np.array(lines)[selections[:, n_id] == 1]]
                else:
                    lorg_is = pd.Series(lorg_is[:, n_id], index=self.timesteps['initial'])
                try:
                    original_line = lorg_is.loc[day].reset_index(drop=True).div(lorg_allowed)
                except TypeError:  # If lorg_allowed is DataFrame instead of Series
                    original_line = lorg_is.loc[day].reset_index(drop=True).div(lorg_allowed.iloc[:, 0])
                ax1_2.plot(original_line.interpolate().bfill(),
                           color=sns.color_palette()[color], linestyle=':', alpha=.6)
            except AttributeError:
                pass

        ax_array[1].set_ylabel('Utilization Voltage [p.u.]')
        ax1_2.set_ylabel('Utilization Line/Transformer [p.u.]')
        ax_array[1].set_xlabel('Time steps')
        ax_array[1].set_xticks(ticks)
        ax_array[1].set_xticklabels(['{}.{}.{}'.format(x.day, x.month, str(x.year)[2:]) for
                                     x in np.unique(day.date.tolist())])
        ax1_2.grid(None)

        labels = ['Transformer load', 'Line load', 'Voltage - MV side', 'Voltage - LV side', 'Original']
        styles = ['-', '-', '-', '-', ':']
        colors = [sns.color_palette()[0], sns.color_palette()[1],
                  sns.color_palette()[2], sns.color_palette()[3], 'black']
        lines = [Line2D([0], [0], color=colors[i], linestyle=styles[i]) for i in range(len(labels))]
        ax_array[1].legend(lines, labels, loc='upper right', prop={'size': 10})

        try:
            fig.tight_layout(pad=1)
        except ValueError:
            pass
        if save_plot:
            plt.savefig(save_file, dpi=800)
        if plotting:
            plt.show()
        plt.close()
        logger.info('Node plot created for node {}, original = {}.'.format(node, original))

    def get_loops(self, inputs):
        import re
        import networkx as nx

        # Get HVMV node and all MVLV nodes
        hv_node = self.edisgo.network.mv_grid.graph.nodes()[
            int(np.where(np.array(
                [repr(x) for x in self.edisgo.network.mv_grid.graph.nodes()]) ==
                         'MVStation_{}'.format(inputs.edisgo_grid))[0])]
        paths = []
        for node in self.edisgo.network.mv_grid.graph.nodes():  # Iterate through all nodes
            path = nx.shortest_path(self.edisgo.network.mv_grid.graph, hv_node, node)  # Get path to node
            path_new = []  # Path to node as list of all n_name on the path
            for node2 in path:  # Iterate through path to node to find nodes with generators (?)
                try:
                    _ = node2.transformers
                    path_new.append(node2)  # Include nodes that have MVLV transformers
                except AttributeError:
                    try:
                        _ = node2.type
                        path_new.append(node2)  # Include nodes that have RES without transformer
                    except AttributeError:
                        try:
                            _ = node2.consumption
                            path_new.append(node2)  # Include nodes that have RES without transformer
                        except AttributeError:
                            pass  # Exclude BranchTees from path

            new = []
            for n in path_new:  # Get n_names for all nodes in path
                if 'Load_aggregated' in repr(n) or 'GeneratorFluctuating' in repr(n):
                    name = repr(n)
                else:
                    name = re.split('_', repr(n))[-1]
                new.append(name)
            paths.append(new)  # Gather the paths for each node
        self.loops = [np.array(x) for x in set(tuple(x) for x in paths)]

    def get_lines_nodes_path(self, inputs):
        import re
        import networkx as nx

        # Preparation, can be done when grid is initiated
        hv_node = self.edisgo.network.mv_grid.graph.nodes()[
            int(np.where(np.array(
                [repr(x) for x in self.edisgo.network.mv_grid.graph.nodes()]) ==
                         'MVStation_{}'.format(inputs.edisgo_grid))[0])]

        # Get the path from each node to HV/MV node and save lines on this path in path_lines
        path_lines = []
        n_names = []
        n_id = []
        for n in self.edisgo.network.mv_grid.graph.nodes():
            if 'Load_aggregated' in repr(n) or 'GeneratorFluctuating' in repr(n):
                name = repr(n)
            else:
                name = re.split('_', repr(n))[-1]

            if name in np.array(self.nodes['node']) and name not in n_names:
                n_names.append(name)
                n_id.append(int(np.where(np.array(self.nodes['node']) == name)[0]))

                path = nx.shortest_path(self.edisgo.network.mv_grid.graph, hv_node, n)
                lines_n_names = []
                for nn in range(len(path) - 1):
                    lines_n_names.append(repr(self.edisgo.network.mv_grid.graph.get_edge_data(path[nn],
                                                                                              path[nn + 1])['line']))
                path_lines.append(lines_n_names)

        # For each line select the nodes that are "behind" that line by going through all paths and if the line is part
        # of the path then get the node to which the path leads
        lines_original = list(self.edisgo.network.mv_grid.graph.lines())
        lines_repr = []
        line_id = 0
        # List of lists of nodes that are connected to a specific line
        selections = np.zeros([len(lines_original), len(n_names)])
        for line in lines_original:
            line = repr(line['line'])
            s = [n if line in path else None for (path, n) in zip(path_lines, n_id)]

            # Convert list of selection s to array of selection and save as selections
            selections[line_id, [x for x in s if x is not None]] = 1

            lines_repr.append(line)
            line_id += 1

        self.line_selections = (lines_repr, selections)

    def nodes_to_lines(self, transformer_loads):
        """Converts an array with loads for all node to an array with loads for all lines, i.e.
        for each line the loads from each node that passed energy through that line are summed up,
        relevant path between node and the HV/MV node is regarded"""

        # Selections: axis0 = lines, axis1 = nodes; shows which nodes are relevant for which lines
        lines, selections = self.line_selections
        lines_nodes = np.multiply(selections, transformer_loads)  # ccpernode at nodes of the line
        line_s = np.nansum(lines_nodes, axis=1) / cfg.efficiency_grid

        return line_s

    # @do_profile(follow=[])
    def lines_to_node(self, distribution_key, line_load):
        """Converts an array with data for all lines to an array with data for all node, i.e.
        depending on how much could be charged at all nodes that are relevant for a line,
        the load of a line is distributed across all nodes that influence a given line based on the
        distribution key"""

        # Selections: axis0 = lines, axis1 = nodes; shows which nodes are relevant for which lines
        lines, selections = self.line_selections
        node_s = np.multiply(selections, distribution_key[None, :])  # ccpernode at nodes of the line
        # sum_node_s = np.nansum(node_s, axis=1)[:, None]
        sum_node_s = np.sum(node_s, axis=1)[:, None]
        node_s = np.divide(node_s, sum_node_s)  # share across all nodes
        node_s = np.multiply(node_s, line_load[:, None])
        node_s[node_s == 0] = np.nan  # Replace 0 (cases were ccpernode is zero) with NaN to find minima
                                      # that are different from 0
        node_s = np.nanmin(node_s, axis=0)
        node_s[np.isnan(node_s)] = 0  # Bring NaNs back to Zero

        return node_s

    def get_grid_problems(self, save_results=cfg.save_results, get_gridproblems=True, get_nodes=False,
                          get_timesteps=False):

        def get_problem(df, problems_count, problems_times, problems_nodes):
            if get_gridproblems:
                problems_count += df.shape[0]
            if get_timesteps:
                problems_times = np.concatenate((problems_times, np.array([date.date() for date in df.time_index])))
            if get_nodes:
                problems_nodes += list(df.index)

            return problems_count, problems_times, problems_nodes

        counts = 0
        times = np.array([], dtype='datetime64[ns]')
        nodes = []

        try:
            for item in self.problems_post:
                try:
                    counts, times, nodes = get_problem(self.problems_post[item], counts, times, nodes)
                except (AttributeError, TypeError):  # If not a DataFrame, but a list or a dict
                    for sub_item in self.problems_post[item]:
                        try:
                            counts, times, nodes = get_problem(sub_item, counts, times, nodes)
                        except AttributeError:  # If not a list, but a dict
                            try:
                                counts, times, nodes = get_problem(self.problems_post[item][sub_item],
                                                                   counts, times, nodes)
                            except TypeError:  # If a list of dicts
                                for sub_sub_item in sub_item:
                                    counts, times, nodes = get_problem(sub_item[sub_sub_item], counts, times, nodes)
            if counts == 0:  # Only save outputs where additional grid problems occur
                save_results = False
        except (AttributeError, TypeError) as e:  # If error occurs at least data can be saved
            counts, nodes, times = e, e, e

        if get_gridproblems:
            return counts, save_results
        elif get_nodes:
            return nodes
        elif get_timesteps:
            return times

    def get_reinforcement_timesteps(self, inputs, ppev_load, test_run=False):
        """Select timesteps from a multitude of extreme load values:
        - Min/Max per node for P and Q loads
        - Min/Max per node for P and Q generation
        - Min/Max per node for S of generation and loads including PEV, individually and combined
        - Min/Max sum across all nodes for P and Q loads and generation,
            and S of generation and loads including PEV
        - Min/Max for all lines to a node for S of generation and loads including PEV
        """

        # Without last 3 where PEV are always fully charged
        loads_t_p_set = self.edisgo.network.pypsa.loads_t.p_set.iloc[:-3, :]
        loads_t_q_set = self.edisgo.network.pypsa.loads_t.q_set.iloc[:-3, :]
        genrs_t_p_set = self.edisgo.network.pypsa.generators_t.p_set.iloc[:-3, :]
        genrs_t_q_set = self.edisgo.network.pypsa.generators_t.q_set.iloc[:-3, :]
        try:
            pd_load = self.pd_load[:-3, :]
            qd_load = self.pd_load[:-3, :]
            pp_load = self.pp_load[:-3, :]
            qp_load = self.qp_load[:-3, :]
        except TypeError:
            pd_load = self.pd_load.iloc[:-3, :].values
            qd_load = self.pd_load.iloc[:-3, :].values
            pp_load = self.pp_load.iloc[:-3, :].values
            qp_load = self.qp_load.iloc[:-3, :].values
        try:
            ppev_load = ppev_load[:-3, :]
        except TypeError:
            ppev_load = ppev_load.iloc[:-3, :].values

        if inputs.selected_steps:
            newindex = np.array([])  # Initialize array of timeteps
            for node in self.edisgo.network.pypsa.loads_t.p_set.columns:  # Get all extreme values of p & q for load
                newindex = np.append(newindex,
                                     [loads_t_p_set.loc[:, node].argmax(),
                                      loads_t_q_set.loc[:, node].argmax(),
                                      loads_t_p_set.loc[:, node].argmin(),
                                      loads_t_q_set.loc[:, node].argmin()
                                      ])
            gen_nodes = self.edisgo.network.pypsa.generators_t.p_set.columns
            try:
                gen_nodes = gen_nodes.drop('Generator_slack')
            except ValueError:
                pass
            for node in gen_nodes:  # Get all extreme values of p & q for gen
                newindex = np.append(newindex,
                                     [genrs_t_p_set.loc[:, node].argmax(),
                                      genrs_t_q_set.loc[:, node].argmax(),
                                      genrs_t_p_set.loc[:, node].argmin(),
                                      genrs_t_q_set.loc[:, node].argmin()
                                      ])

            load = pd_load if ppev_load is None else pd_load + ppev_load
            sload = calc_apparent_power(load, qd_load)
            sgener = calc_apparent_power(pp_load, qp_load)

            for n_id in range(sload.shape[1]):  # Get all extreme values of apparent power for load and gen
                newindex = np.append(newindex,
                                     [inputs.index[sload[:, n_id].argmax()],
                                      inputs.index[sload[:, n_id].argmin()],
                                      inputs.index[sgener[:, n_id].argmax()],
                                      inputs.index[sgener[:, n_id].argmin()],
                                      inputs.index[(sload - sgener)[:, n_id].argmax()],
                                      inputs.index[(sload - sgener)[:, n_id].argmin()]
                                      ])
            newindex = np.append(newindex,  # Get all extreme values for sum of all load across nodes
                                 [loads_t_p_set.sum(1).argmax(),
                                  loads_t_q_set.sum(1).argmax(),
                                  loads_t_p_set.sum(1).argmin(),
                                  loads_t_q_set.sum(1).argmin(),
                                  genrs_t_p_set.sum(1).argmax(),
                                  genrs_t_q_set.sum(1).argmax(),
                                  genrs_t_p_set.sum(1).argmin(),
                                  genrs_t_q_set.sum(1).argmin(),
                                  inputs.index[sload.sum(1).argmax()],
                                  inputs.index[sload.sum(1).argmin()],
                                  inputs.index[sgener.sum(1).argmax()],
                                  inputs.index[sgener.sum(1).argmin()]
                                  ])
            for loop in self.loops:
                l_id = np.where(np.isin(self.nodes['node'], loop))[0]
                try:
                    newindex = np.append(newindex,
                                         [inputs.index[sload[:, l_id].sum(1).argmax()],
                                          inputs.index[sload[:, l_id].sum(1).argmin()],
                                          inputs.index[sgener[:, l_id].sum(1).argmax()],
                                          inputs.index[sgener[:, l_id].sum(1).argmin()],
                                          inputs.index[(sload - sgener)[:, l_id].sum(1).argmax()],
                                          inputs.index[(sload - sgener)[:, l_id].sum(1).argmin()]
                                          ])
                except KeyError:
                    pass

            newindex = newindex[newindex < inputs.index[-3]]  # Checks that timesteps are not last three steps
            newdatetimeindex = self.edisgo.network.pypsa.loads_t.p_set.loc[newindex, :].index
            if not test_run:  # Random sample will be added for all Sets of one test iteration together
                self.timesteps['initial'] = newdatetimeindex.append(self.random_sample.index)
            else:
                self.timesteps['initial'] = newdatetimeindex
            self.timesteps['initial'] = self.timesteps['initial'].unique().sort_values()
        else:
            # Without last 3 where PEV are always fully charged
            self.timesteps['initial'] = inputs.dates_represented_index

    def get_loads_dict(self, getting=False, setting=False, loads_dict=None, set_p_set_only=False, **kwargs):
        if setting and loads_dict is not None:
            # Adapt RES share if necessary for scenario
            try:
                res_share = kwargs['inputs'].res_share
            except AttributeError:
                res_share = 1

            if set_p_set_only:
                self.edisgo.network.pypsa.loads_t.p_set = loads_dict['loads_t']['p_set']
                self.edisgo.network.pypsa.loads_t.q_set = loads_dict['loads_t']['q_set']
                self.edisgo.network.pypsa.generators_t.p_set = loads_dict['generators_t']['p_set']
                self.edisgo.network.pypsa.generators_t.q_set = loads_dict['generators_t']['q_set']
            else:
                self.edisgo.network.pypsa.loads_t = loads_dict['loads_t']
                self.edisgo.network.pypsa.generators_t = loads_dict['generators_t']
                self.pd_load = loads_dict['pd_load']
                self.qd_load = loads_dict['qd_load']
                self.pp_load = loads_dict['pp_load'] * res_share
                self.qp_load = loads_dict['qp_load'] * res_share
                self.pcurtail = loads_dict['pcurtail'] * res_share
                self.qcurtail = loads_dict['qcurtail'] * res_share

            if res_share != 1:
                for key in self.edisgo.network.pypsa.generators_t.keys():
                    self.edisgo.network.pypsa.generators_t[key] *= res_share
        elif getting:
            output = {'loads_t': self.edisgo.network.pypsa.loads_t,
                      'generators_t': self.edisgo.network.pypsa.generators_t,
                      'pd_load': self.pd_load,
                      'qd_load': self.qd_load,
                      'pp_load': self.pp_load,
                      'qp_load': self.qp_load,
                      'pcurtail': self.pcurtail,
                      'qcurtail': self.qcurtail}

            return output
        else:
            logger.error('Loads dict not defined, can not set new loads')

    def plot_line_location(self, lines_selection=[9, 13, 27, 32, 44, 45, 46, 48]):
        from edisgo.tools import plots

        lines = [x['line'] for x in
                 np.array(list(self.edisgo.network.mv_grid.graph.lines()))[lines_selection]]
        grid_expansion_costs = \
            pd.DataFrame([],
                         columns=self.edisgo.network.results.grid_expansion_costs.reset_index().columns,
                         index=range(len(lines)))

        number = 0
        for line in lines:
            grid_expansion_costs.loc[number] = [line, 1, line, 1, 999, 'NA2XS2Y 3x1x240 RM/25', 'mv']
            number += 1
        grid_expansion_costs['index'] = \
            grid_expansion_costs['index'].apply(lambda _: repr(_))
        grid_expansion_costs.set_index('index', inplace=True)
        plots.mv_grid_topology(
            self.edisgo.network.pypsa, self.edisgo.network.config,
            line_color='expansion_costs',
            grid_expansion_costs=grid_expansion_costs,
        )


def sample_scenario(inputs, plot=False):
    """Bootstraps load and generation curves to generate a new random sample
    Sample RES from Sommer = 01.06. – 31.08., Winter = 01.12. – 29.02.,
                    Frühling = 01.03. – 31.05., Herbst = 01.09. – 30.11.
    Sample demand from Sommer = 01.06. – 31.08., Winter = 01.12. – 29.02.,
                       Übergangszeit = 01.03. – 31.05 & 01.09. – 30.11.
    """
    import random
    from Functions import cluster_functions as clf

    if plot:
        import seaborn as sns
        from scipy import stats
        import matplotlib.pyplot as plt

        # Correlation between RES and demand
        grid = get_grid(inputs)
        load = grid.edisgo.network.pypsa.loads_t.p_set.sum(1)
        res = grid.edisgo.network.pypsa.generators_t.p_set.sum(1)
        res.name, load.name = 'res', 'demand'
        data = pd.concat([res, load], axis=1)

        cfg.set_design(palette_size=8)
        x_plots, y_plots = 4, 3
        fig, axes = plt.subplots(x_plots, y_plots)
        out = pd.DataFrame([],
                           index=list(range(12)),
                           columns=['coefficient', 'pvalue'])
        m = 0
        for x in range(x_plots):
            for y in range(y_plots):
                data_analyse = data.iloc[m * 1460:(m + 1) * 1460, :].resample(
                    'D').max()  # Select month and aggregate days
                data_analyse = data_analyse[(np.abs(stats.zscore(data_analyse)) < 3).all(axis=1)]  # Remove outlier
                pvalues = clf.calculate_pvalues(data_analyse, roundby=20)
                out.loc[m, 'coefficient'] = data_analyse.loc[:, 'demand'].corr(data_analyse.loc[:, 'res'])
                out.loc[m, 'pvalue'] = pvalues.loc['res', 'demand']
                axis = axes[x, y]
                sns.regplot(x='demand', y='res', data=data_analyse, ax=axis)
                axis.set_title('month = {}, corr = {}, p = {}'.format(m, out.loc[m, 'coefficient'].round(4),
                                                                      out.loc[m, 'pvalue'].round(4)))
                m += 1
        fig.tight_layout(pad=2)
        plt.savefig('Energy_correlation_{}.png'.format(grid.edisgo.network.id))
        logger.info('Correlation between energy generation and load per day for each month: \n{}'.format(out))

    if not os.path.isfile(inputs.grid_file_scenario):
        logger.warning('Generating load scenario for {} on process {} for ... {}'.format(
            inputs.episode, os.getpid(), inputs.grid_file_scenario[-50:]))

        # Correlation between RES and demand
        grid = get_grid(inputs)
        load = grid.edisgo.network.pypsa.loads_t.p_set.sum(1)
        res = grid.edisgo.network.pypsa.generators_t.p_set.sum(1)
        res.name, load.name = 'res', 'demand'
        data = pd.concat([res, load], axis=1)

        # np.random.seed(inputs.episode)
        days = data.resample('D').sum().index

        # Select seasons
        seasons = {'spring': data.loc[(data.index.month >= 3) & (data.index.month < 6), :].resample('D').sum().index,
                   'summer': data.loc[(data.index.month >= 6) & (data.index.month < 9), :].resample('D').sum().index,
                   'fall': data.loc[(data.index.month >= 9) & (data.index.month < 12), :].resample('D').sum().index,
                   'winter': None, 'inbetween': None}
        seasons['winter'] = days.drop(seasons['spring']).drop(seasons['summer']).drop(seasons['fall'])
        seasons['inbetween'] = seasons['spring'].append(seasons['fall'])

        # Select weekdays
        season_dict = {'saturday': None, 'sunday': None, 'weekday': None}
        seasons_days = {'spring': copy.deepcopy(season_dict), 'summer': copy.deepcopy(season_dict),
                        'fall': copy.deepcopy(season_dict), 'winter': copy.deepcopy(season_dict),
                        'inbetween': copy.deepcopy(season_dict)}
        for season in seasons.keys():
            seasons_days[season]['saturday'] = seasons[season][seasons[season].weekday == 5]
            seasons_days[season]['sunday'] = seasons[season][seasons[season].weekday == 6]
            seasons_days[season]['weekday'] = seasons[season].drop(seasons_days[season]['saturday']
                                                                   ).drop(seasons_days[season]['sunday'])

        # Sample from seasons
        data = copy.deepcopy(grid.edisgo.network.pypsa.loads_t)
        for load in ['p_set', 'q_set', 'p', 'q']:
            data[load] = pd.DataFrame()

        data_generators, data_loads = copy.deepcopy(data), copy.deepcopy(data)
        for day in days:
            # Select the correct season
            season_generators, season_loads = None, None
            for season_selection in ['inbetween', 'summer', 'winter']:
                if day in seasons[season_selection]:
                    season_generators = season_selection
            for season_selection in ['spring', 'summer', 'fall', 'winter']:
                if day in seasons[season_selection]:
                    season_loads = season_selection

            # Select weekday
            if day.weekday() == 5:  # Saturday
                weekday_loads = 'saturday'
            elif day.weekday() == 6:  # Sunday
                weekday_loads = 'sunday'
            else:  # Weekday
                weekday_loads = 'weekday'

            # Get full loads for the day from pypsa object
            for data, scenarios, pypsa in \
                    [(data_generators, seasons[season_generators], grid.edisgo.network.pypsa.generators_t),
                     (data_loads, seasons_days[season_loads][weekday_loads], grid.edisgo.network.pypsa.loads_t)]:
                scenario = random.choice(scenarios)
                data['p_set'] = pd.concat([data['p_set'], pypsa['p_set'][pypsa['p_set'].index.date == scenario.date()]])
                data['q_set'] = pd.concat([data['q_set'], pypsa['q_set'][pypsa['q_set'].index.date == scenario.date()]])

        # Reset to original index
        for data, pypsa in [(data_generators, grid.edisgo.network.pypsa.loads_t),
                            (data_loads, grid.edisgo.network.pypsa.loads_t)]:
            data['p_set'].index = pypsa['p_set'].index
            data['q_set'].index = pypsa['q_set'].index
            data['p'] = copy.deepcopy(data['p_set'])
            data['q'] = copy.deepcopy(data['q_set'])

        grid.edisgo.network.pypsa.loads_t = data_loads
        grid.edisgo.network.pypsa.generators_t = data_generators

        # Update all loads required in get_loads_dict and save scenario
        grid.update_loads_and_consumption(inputs)
        with open(inputs.grid_file_scenario, 'wb') as f:
            dill.dump(grid.get_loads_dict(getting=True), f)
    else:
        logger.warning('Load scenario for {} already available in  ... {}'.format(
            inputs.episode, inputs.grid_file_scenario[-50:]))


def get_set_of_loads(inputs_dict, max_parallel=10):
    import multiprocessing as mp
    from functools import partial

    logger.info('Generating load scenarios for the following inputs: \n{}'.format(inputs_dict))

    # Calculate sets of loads
    pool = mp.Pool(processes=min(max_parallel, len(inputs_dict)),
                   maxtasksperchild=1)  # mp.cpu_count())  # Spawn processes
    func = partial(sample_scenario)
    pool.map(func, list(inputs_dict.values()), chunksize=1)
    pool.close()
    pool.join()

    return inputs_dict


def calc_apparent_power(true, reactive):  # Apparent Power is squareroot(True power2 + Reactive Power2)
    apparent = (true ** 2 + reactive ** 2) ** 0.5  # kVA
    flow = (true / abs(true))
    flow.fill(1)  # * (-1) if reverse load flow
    apparent = flow * apparent

    return apparent


def get_dingo_file(inputs, file):
    import time
    from egoio.tools import db
    from ding0.core import NetworkDing0
    from ding0.tools.results import save_nd_to_pickle
    from sqlalchemy.orm import sessionmaker
    from oedialect.engine import ConnectionException
    from sqlalchemy.exc import DBAPIError

    failed_file = '{}/Grids/Grid_[{}]_FAILED.pkl'.format(cfg.parent_folder, inputs.edisgo_grid)
    e = ''
    for _ in range(cfg.oedb_acces_tries):
        try:
            # database connection/ session
            engine = db.connection(section='oedb')
            session = sessionmaker(bind=engine)()

            nd = NetworkDing0(name='network')  # instantiate new ding0 network object
            nd.run_ding0(session=session,
                         mv_grid_districts_no=[inputs.edisgo_grid],
                         export_figures=False)  # run DING0 on selected MV Grid District

            # export grid to file (pickle)
            save_nd_to_pickle(nd,
                              filename=file)
            return
        except (ConnectionException, DBAPIError) as e:
            wait = np.random.randint(cfg.waittime)
            logger.info('Timeseries import from OEDB FAILED, waiting {} seconds\n'
                        'ERROR: {}'.format(wait, e))
            time.sleep(wait)
        except AttributeError as e:
            with open(failed_file, 'wb') as f:
                dill.dump(e, f)
            raise AssertionError('Gettting ding0 file has failed due to the following: {}'.format(e))

    raise AssertionError('Gettting ding0 file has failed {} times due to the following: {}'.format(
        cfg.oedb_acces_tries, e))


def get_initial_edisgo_grid(inputs, file_edisgo, save=True):
    from edisgo import EDisGo
    from edisgo.grid.network import Results
    from Functions import loadflow_functions as lff
    # https://www.uni-flensburg.de/fileadmin/content/abteilungen/industrial/dokumente/downloads/veroeffentlichungen/forschungsergebnisse/20190426endbericht-openego-fkz0325881-final.pdf

    if os.path.isfile(file_edisgo):
        logger.warning('Edisgo object for grid {} exists already'.format(
            inputs.edisgo_grid))
        with open(file_edisgo, 'rb') as f:
            edisgo = dill.load(f)
        return edisgo

    ding0_grid = '{}/Grids/Grid_[{}].pkl'.format(cfg.parent_folder, inputs.edisgo_grid)
    if not os.path.isfile(ding0_grid):
        get_dingo_file(inputs, ding0_grid)
    try:
        edisgo = EDisGo(ding0_grid=ding0_grid,
                        worst_case_analysis='worst-case')
        edisgo = lff.reinforce(edisgo, inputs, mode='mv')
        logger.warning('Initial expansion cost for grid {}: {}'.format(
            inputs.edisgo_grid, edisgo.network.results.grid_expansion_costs.total_costs.sum()))  # [kEUR]
        edisgo.network.results = Results(edisgo.network)  # Reset initial expansion cost
        edisgo.network.pypsa = None
        if save:
            with open(file_edisgo, 'wb') as f:
                dill.dump(edisgo, f)

        logger.info('Edisgo object for grid {} generated'.format(
            inputs.edisgo_grid))
        return edisgo
    except (IndexError, KeyError) as e:
        raise AssertionError('Error in getting initial grid: {}: {}'.format(type(e), e))


def get_grid(inputs, return_bool=True, get_res=True, file_gridobject=None, scenario_only=False, first=False):
    """Initialize grid including non-PEV loads for the full year"""

    def import_grid():
        try:
            with open(file_gridobject, 'rb') as f:
                grid_object = dill.load(f)
        except (TypeError, EOFError):
            with open(file_gridobject, 'rb') as f:
                grid_object = dill.load(f)[0]

        return grid_object

    # Get distribution grid
    if file_gridobject is None:
        file_gridobject = inputs.grid_file
    logger.debug('file_gridobject = {}'.format(file_gridobject[-50:]))

    file_gridobjecttemp = '{}_temp.dill'.format(file_gridobject[:-4])
    try:
        if not os.path.isfile(file_gridobject):
            logger.info('Setting up the grid {} for year {}'.format(inputs.edisgo_grid, inputs.year))
            grid = DistributionGrid(inputs.year)
            error = grid.setup_edisgo_grid(inputs,
                                           file_gridobjecttemp,
                                           get_res=get_res,
                                           full_year=True,
                                           scenario_only=scenario_only)
            grid.get_loops(inputs)
            grid.get_lines_nodes_path(inputs)
            # grid.get_state_dependent_quantities(inputs)  # Needed?
            grid.transformer_sallowed = grid.get_aggregated_overloads(inputs,
                                                                      transformer=True,
                                                                      return_allowed=True,
                                                                      incl_hvmv=False).max(0)
            grid.line_sallowed = grid.get_aggregated_overloads(inputs,
                                                               line=True,
                                                               return_allowed=True).values
            grid.get_equipment()  # Write all lines in the grid in one DataFrame
            grid.random_sample = cfg.timestep_samples(inputs, grid)
            # Test if pkw_number is saved in grid object
            if grid.pkw_number == 0:
                with open('{}/Inputs/grid_pkw_numbers.pkl'.format(cfg.parent_folder), 'rb') as f:
                    pkw_numbers = dill.load(f)
                grid.pkw_number = pkw_numbers[pkw_numbers.grid_id == inputs.edisgo_grid].PKW_gridPLZ.sum()

            with open(file_gridobject, 'wb') as f:
                dill.dump([grid, error], f)
            if cfg.remove_grid_temp:
                try:
                    os.remove(file_gridobjecttemp)
                except FileNotFoundError:
                    pass
            logger.info('Grid {} for year {} generated and saved'.format(inputs.edisgo_grid, inputs.year))
        else:
            # Import grid while checking for MemoryError, wait and repeat if needed
            import time
            try_number = 0
            while try_number < cfg.tries_if_memoryerror:
                try:
                    grid = import_grid()
                    break
                except MemoryError as e:
                    rand_waittime = np.random.rand() * cfg.waittime
                    logger.error('MemoryError ({}) when importing grid: {}, retry after {} seconds'.format(
                        type(e), e, rand_waittime))
                    time.sleep(rand_waittime)
                    try_number += 1
                    raise MemoryError('MemoryError ({}) when importing grid: {}, tried {} times'.format(
                        type(e), e, try_number))

            logger.debug('File {} imported as grid'.format(file_gridobject[-50:]))

            if first:
                original_nodes = len(grid.nodes['node'])
                original_consumption = grid.consumption.sum().sum()
                grid.update_loads_and_consumption(inputs)
                original_line_sallowed = grid.line_sallowed.sum()
                grid.line_sallowed = grid.get_aggregated_overloads(inputs,
                                                                   line=True,
                                                                   return_allowed=True).values
                if (grid.pkw_number == 0 or
                        grid.random_sample.shape != (inputs.sample_size, 2) or
                        grid.transformer_sallowed.max() < 99999 or
                        original_nodes != len(grid.nodes['node']) or
                        original_consumption != grid.consumption.sum().sum() or
                        original_line_sallowed != grid.line_sallowed.sum()):
                    if grid.pkw_number == 0:  # Test if pkw_number is saved in grid object
                        with open('{}/Inputs/grid_pkw_numbers.pkl'.format(cfg.parent_folder), 'rb') as f:
                            pkw_numbers = dill.load(f)
                        logger.warning('!!! - PKW numbers == {}, is being updated to {}'.format(
                            grid.pkw_number, pkw_numbers))
                        grid.pkw_number = pkw_numbers[pkw_numbers.grid_id == inputs.edisgo_grid].PKW_gridPLZ.sum()

                    if grid.random_sample.shape != (inputs.sample_size, 2):  # Test if sample time periods size is correct
                        logger.warning('!!! - Number of sample timesteps = {}, is being updated to {}'.format(
                            grid.random_sample.shape, (inputs.sample_size, 2)))
                        grid.random_sample = cfg.timestep_samples(inputs, grid)

                    # Test if new limits for direct connections to MV grid are implemented
                    if grid.transformer_sallowed.max() < 99999:
                        grid.transformer_sallowed = grid.get_aggregated_overloads(inputs,
                                                                                  transformer=True,
                                                                                  return_allowed=True,
                                                                                  incl_hvmv=False).max(0)
                    if (original_nodes != len(grid.nodes['node']) or
                            original_consumption != grid.consumption.sum().sum() or
                            original_line_sallowed != grid.line_sallowed.sum()):
                        grid.get_loops(inputs)
                        grid.get_lines_nodes_path(inputs)
                        grid.transformer_sallowed = grid.get_aggregated_overloads(inputs,
                                                                                  transformer=True,
                                                                                  return_allowed=True,
                                                                                  incl_hvmv=False).max(0)
                        grid.line_sallowed = grid.get_aggregated_overloads(inputs,
                                                                           line=True,
                                                                           return_allowed=True).values
                    if file_gridobject == inputs.grid_file:  # Only save if concearning the original grid
                        with open(file_gridobject, 'wb') as f:
                            dill.dump(grid, f)

        if return_bool:
            return grid
    except (TypeError, KeyError) as error:
        with open('{}_FAILED.pkl'.format(file_gridobject[:-4]), 'wb') as f:
            dill.dump(error, f)
        raise AssertionError('Grid {} for year {} could not be generated, due to: {}: {}'.format(
                     inputs.edisgo_grid, inputs.year, type(error), error))


def get_optimized_grid(inputs, test_number=None, loads_dict=None, name='g2vucc_fullres',
                       original_grid=True, combined_expansion=True, set_p_set_only=False):
    """Prepares grid selection depending on test (UCC or G2VUCC) and which iteration of test (first it UCC gets
    original grid and G2VUCC grid from best test run) all subsequent iterations the respective grid from the last
    iteration is used"""
    from glob import glob
    from edisgo.grid.network import Results

    if (original_grid and test_number == 0) or (name == 'base_ucc') or (name == 'g2vucc_fullres'):
        logger.warning('Getting original grid object without any expansions')
        grid_file = None
    else:
        if test_number == 0:
            if name == 'test_ucc':  # Getting worst case ucc_base as used for optimization
                grid_files = '{}_base_uccout_e*_p*_c*_grid.dill'.format(inputs.out_file)
            elif name == 'test_best_g2vucc':  # Getting file from best optimization run
                cost_best = pd.read_csv('{}_cost_best.csv'.format(inputs.out_file), index_col=0
                                        ).query('type == "g2vucc_fullres"').total_cost.min()
                grid_files = '{}_out_e*_p*_c{}_grid.dill'.format(inputs.out_file, cost_best)
            else:
                raise FileNotFoundError(
                    'No matching grid file found for {} and test iteration {}'.format(name, test_number))
        else:  # Getting file from previous test iteration
            if combined_expansion:
                grid_files = '{}_{}_t{}_p*_c*_grid.dill'.format(inputs.out_file, name, test_number - 1)
            else:
                grid_files = '{}_{}out_e{}_p*_c*_grid.dill'.format(inputs.out_file, name, inputs.episode - 1)

        logger.debug('Searching for the following grid files: {}'.format(grid_files))
        grid_file = glob(grid_files)[0]
        logger.info('Getting grid object from file ... {}'.format(grid_file[-50:]))

    grid = get_grid(inputs, file_gridobject=grid_file)

    # Reset grid loads to selected scenario
    if not loads_dict:
        logger.info('Utilizing the loads from the following file: {}'.format(inputs.grid_file_scenario[-50:]))
        with open(inputs.grid_file_scenario, 'rb') as f:
            loads_dict = dill.load(f)

    kwargs = {"inputs": inputs}
    grid.get_loads_dict(setting=True, loads_dict=loads_dict, set_p_set_only=set_p_set_only, **kwargs)
    grid.edisgo.network.results = Results(grid.edisgo.network)  # Reset initial expansion cost

    if test_number is not None:  # Update lines and nodes path in case existing grid has been expanded
        grid.get_loops(inputs)
        grid.get_lines_nodes_path(inputs)
        grid.transformer_sallowed = grid.get_aggregated_overloads(inputs,
                                                                  transformer=True,
                                                                  return_allowed=True,
                                                                  incl_hvmv=False).max(0)
        grid.line_sallowed = grid.get_aggregated_overloads(inputs,
                                                           line=True,
                                                           return_allowed=True).values

    return grid


def plot_all_nodes_all_days(out_file, inputs=None, grid=None, pevs=None, number_days=None, original=False, nodes=None,
                            all_nodes=True, add_test_expansions=True):
    """Plot situation at node with PEV and curtailment"""
    import re
    from glob import glob
    import matplotlib as mpl

    if inputs is None:
        with open('{}parameters.dill'.format(re.split('out', out_file)[0]), 'rb') as f:
            inputs = dill.load(f)

    if original:
        # Get original grid
        with open(inputs.grid_file, 'rb') as f:
            grid = dill.load(f)
        loads = grid.setup_plot(None, inputs, original=True)
        pevs = None
    else:
        if grid is None:
            try:
                with open('{}_grid.dill'.format(out_file), 'rb') as f:
                    grid = dill.load(f)[0]
            except (EOFError, TypeError):
                with open('{}_grid.dill'.format(out_file), 'rb') as f:
                    grid = dill.load(f)
        if pevs is None:
            try:
                with open('{}_pevs.dill'.format(out_file), 'rb') as f:
                    pevs = dill.load(f)[0]
            except (EOFError, TypeError):
                with open('{}_pevs.dill'.format(out_file), 'rb') as f:
                    pevs = dill.load(f)
        loads, pevs = grid.setup_plot(pevs, inputs)

    try:
        grid.edisgo.plot_mv_grid_expansion_costs(filename='{}_expansioncost_grid.png'.format(out_file))
    except ValueError as e:
        logger.error('Could not plot {} due to {}'.format('{}_expansioncost_grid.png'.format(out_file), e))

    cfg.set_design(palette_size=8)
    mpl.rcParams['agg.path.chunksize'] = 10000

    trafo_sallowed = grid.get_aggregated_overloads(inputs, transformer=True, return_allowed=True, incl_hvmv=False)
    transformer_s = grid.get_aggregated_overloads(inputs, transformer=True, return_is=True, incl_hvmv=False)

    line_sallowed = grid.get_aggregated_overloads(inputs, line=True, return_allowed=True, lf_results=True)
    line_s = grid.get_aggregated_overloads(inputs, line=True, return_is=True)
    # over_v, under_v = grid.get_aggregated_overloads(inputs, voltage=True, return_is=True)

    v_mvside = {'min': None, 'max': None, 'is': None}
    v_lvside = {'min': None, 'max': None, 'is': None}
    for var, level in [[v_mvside, 'mv'], [v_lvside, 'lv']]:
        var['min'], var['max'] = grid.get_aggregated_overloads(inputs,
                                                               voltage=True,
                                                               return_allowed=True,
                                                               network=grid.edisgo.network,
                                                               timesteps=grid.timesteps['final'],
                                                               level=level)
        var['is'] = grid.get_aggregated_overloads(inputs,
                                                  voltage=True,
                                                  return_is=True,
                                                  network=grid.edisgo.network,
                                                  timesteps=grid.timesteps['final'],
                                                  level=level)

    # Get grid states from original case
    trafo_sallowed_original = grid.get_aggregated_overloads(inputs, transformer=True, return_allowed=True,
                                                            original=True, incl_hvmv=False)
    line_sallowed_original = grid.get_aggregated_overloads(inputs, line=True, return_allowed=True, original=True,
                                                           lf_results=True)

    # Select nodes that originally have load or voltage issues, but exclude nodes that get values from G2N
    if nodes is None and not all_nodes:
        # Add nodes where tests are having problems
        if add_test_expansions and 'test' not in out_file:
            try:
                test_file = glob('{}_outtest_*_grid.dill'.format(re.split('out', out_file)[0]))[0]
                with open(test_file, 'rb') as f:
                    grid_test = dill.load(f)
                grids = [grid, grid_test]
            except IndexError:
                grids = [grid]
        else:
            grids = [grid]

        nodes = []
        for g in grids:
            nodes += g.get_grid_problems(get_gridproblems=False, get_nodes=True)

        nodes_id = []
        for x in nodes:
            if 'Station' in repr(x):
                nodes_id.append(str(x.id))
            elif 'Line' in repr(x):
                line_id = np.where(np.array(g.line_selections[0]) == repr(x))[0]
                nodes_id += list(np.array(g.nodes['node'])[g.line_selections[1][line_id][0] == 1])

    max_grid_problems = grid.get_grid_problems(get_gridproblems=False, get_timesteps=True)
    max_steps = loads['sp_load'].shape[0]
    index_full = pd.date_range('2011-01-01',
                               periods=max_steps,
                               freq=str(24 / cfg.t_periods) + 'H')

    for node in set(nodes_id) if not all_nodes else grid.nodes['node']:
        try:
            n_id = np.where(np.array(list(grid.nodes['node'])) == node)[0][0]
            save_file = '{}_Node_{}.png'.format(out_file, node[:40])
            if not os.path.isfile(save_file):
                # Select timesteps
                max_load_timestep = grid.timesteps['initial'][np.argpartition(transformer_s[:, n_id],
                                                                              -number_days)[-number_days:]].date
                date_full = pd.DatetimeIndex([])
                for day in np.unique(np.concatenate([max_grid_problems, max_load_timestep])):
                    date_full = date_full.append(index_full[index_full.date == day])

                # Prepare grid stats
                grid_stats_node = {}
                grid_stats_node['v_mvside_max'] = v_mvside['max']
                grid_stats_node['v_mvside_min'] = v_mvside['min']
                for var, name in zip([trafo_sallowed, line_sallowed, line_s, trafo_sallowed_original,
                                      line_sallowed_original, transformer_s, v_mvside['is'], v_lvside['is'],
                                      v_lvside['max'], v_lvside['min']],
                                     ['trafo_sallowed', 'line_sallowed', 'line_s', 'trafo_sallowed_original',
                                      'line_sallowed_original', 'transformer_s', 'v_mvside_is', 'v_lvside_is',
                                      'v_lvside_max', 'v_lvside_min']):

                    if 'line' in name:  # Select lines that are relevant for a node
                        lines, selections = grid.line_selections
                        var_selected = var.loc[:, np.array(lines)[selections[:, n_id] == 1]]
                    else:  # Select relevant node
                        var_selected = var[:, n_id]

                    # Get data for individual node from data of entire grid or if data from combined grid testing
                    if 'original' in name:  # For allowed values of original grid
                        var_selected_temp = np.array([])
                        it = 0
                        while var_selected_temp.shape[0] < max_steps:
                            if it == 0:
                                var_selected_temp = var_selected
                            else:
                                var_selected_temp = np.concatenate((var_selected_temp, var_selected), axis=0)
                            it += 1
                        var_selected = var_selected_temp
                        index = index_full
                    else:
                        if var_selected.shape[0] == inputs.index.shape[0]:
                            index = inputs.index  # Full year available for optimization runs
                        else:
                            index = grid.timesteps['final']  # Only selected timesteps available for test runs

                    try:
                        grid_stats_node[name] = pd.DataFrame(var_selected, index=index
                                                             ).loc[date_full].reset_index(drop=True)
                    except ValueError as e:
                        print('Error: {}'.format(e))

                # Prepare loads (select node and timesteps)
                loads_node = {}
                for key in loads.keys():
                    try:
                        loads_node[key] = loads[key].loc[date_full, node].reset_index(drop=True)
                    except (AttributeError, IndexError) as e:
                        print('Error: {}'.format(e))

                # Plot
                grid.plot_one_node(save_file, pevs if not original else None, node, n_id, date_full, loads_node,
                                   grid_stats_node, plotting=False, original=original)
        except (KeyError, IndexError) as e:
            logger.info('Node {} not found, can not be plotted due to {}'.format(node, e))
