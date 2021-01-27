import pandas as pd
import numpy as np
import copy
import datetime
from edisgo.flex_opt import check_tech_constraints as checks
from edisgo.flex_opt import reinforce_measures, exceptions
from edisgo.flex_opt.costs import grid_expansion_costs
from edisgo.tools import tools, pypsa_io
from edisgo.grid.tools import assign_mv_feeder_to_nodes
import logging
logger = logging.getLogger(__name__)


def get_line_overloads(network, return_allowed=False, return_is=False):
    grid_level = 'mv'
    s_line = pd.DataFrame()
    s_line_allowed = pd.DataFrame()
    for line in list(network.mv_grid.graph.lines()):
        if return_allowed:
            s_line_allowed_per_case = {
                'feedin_case': (line['line'].type['I_max_th'] * line['line'].type['U_n'] / (3 ** 0.5)
                                ) * line['line'].quantity * network.config[
                                'grid_expansion_load_factors']['{}_feedin_case_line'.format(grid_level)],
                'load_case': (line['line'].type['I_max_th'] * line['line'].type['U_n'] / (3 ** 0.5)
                              ) * line['line'].quantity * network.config[
                              'grid_expansion_load_factors']['{}_load_case_line'.format(grid_level)]}
            # maximum allowed line load in each time step
            s_line_allowed[repr(line['line'])] = network.timeseries.timesteps_load_feedin_case.case.apply(
                lambda _: s_line_allowed_per_case[_])

        if return_is:
            try:
                # check if capacity from pf analysis exceeds allowed maximum capacity
                s_line[repr(line['line'])] = network.results.s_res()[repr(line['line'])]
            except KeyError:
                logger.error('No results for line {} '.format(str(line)) + 'to check overloading.')

    if return_allowed:
        return s_line_allowed
    elif return_is:
        return s_line


def get_line_overloads_i(network, return_allowed=False, return_is=False):
    grid_level = 'mv'
    crit_lines = pd.DataFrame()
    i_line = pd.DataFrame()
    i_line_allowed = pd.DataFrame()
    for line in list(network.mv_grid.graph.lines()):
        i_line_allowed_per_case = {
            'feedin_case': line['line'].type['I_max_th'] * line['line'].quantity * network.config[
                'grid_expansion_load_factors']['{}_feedin_case_line'.format(grid_level)],
            'load_case': line['line'].type['I_max_th'] * line['line'].quantity * network.config[
                'grid_expansion_load_factors']['{}_load_case_line'.format(grid_level)]}

        # maximum allowed line load in each time step
        i_line_allowed[repr(line['line'])] = network.timeseries.timesteps_load_feedin_case.case.apply(
            lambda _: i_line_allowed_per_case[_])

        try:
            # check if current from pf analysis exceeds allowed maximum current
            i_line[repr(line['line'])] = network.results.i_res[repr(line['line'])]
            crit_lines[repr(line['line'])] = (i_line[repr(line['line'])] - i_line_allowed[repr(line['line'])]
                                              ).div(i_line_allowed[repr(line['line'])])
        except KeyError:
            logger.error('No results for line {} '.format(str(line)) + 'to check overloading.')

    if return_allowed:
        return i_line_allowed
    elif return_is:
        return i_line
    else:
        return crit_lines


def get_transformer_overloads(network, station, grid_level, return_allowed=False, return_is=False):
    """ edisgo.flex_opt.check_tech_constraints.hv_mv_station_load(network)[source]
        edisgo.flex_opt.check_tech_constraints.mv_lv_station_load(network)[source]"""
    from edisgo.grid.components import LVStation

    crit_stations = pd.DataFrame()
    # maximum allowed apparent power of station for feed-in and load case
    s_station = sum([_.type.S_nom for _ in station.transformers])
    s_station_allowed_per_case = {
        'feedin_case': s_station * network.config['grid_expansion_load_factors'][
            '{}_feedin_case_transformer'.format(grid_level)],
        'load_case': s_station * network.config['grid_expansion_load_factors'][
            '{}_load_case_transformer'.format(grid_level)]}

    # maximum allowed apparent power of station in each time step
    s_station_allowed = network.timeseries.timesteps_load_feedin_case.case.apply(
        lambda _: s_station_allowed_per_case[_])
    if return_allowed:
        return s_station_allowed.rename(repr(station))

    try:
        if isinstance(station, LVStation):
            s_station_pfa = network.results.s_res(station.transformers).sum(axis=1)
        else:
            s_station_pfa = network.results.s_res([station]).iloc[:, 0]
        crit_stations[repr(station)] = (s_station_pfa - s_station_allowed).div(s_station_allowed)
    except KeyError:
        logger.error('No results for {} station to check overloading.'.format(grid_level.upper()))
        s_station_pfa = None

    if return_is:
        return s_station_pfa.rename(repr(station))
    else:
        return crit_stations


def get_voltage_is(network, level='mv'):
    """ Extracting voltages from edisgo object. Returns both as DataFrames with Index = Timesteps
    and Columns = MV Nodes
    Sources:
    - edisgo.flex_opt.check_tech_constraints.mv_voltage_deviation(network, voltage_levels='mv_lv')[source]
    - edisgo.flex_opt.check_tech_constraints.check_ten_percent_voltage_deviation(network)[source]"""

    nodes = network.mv_grid.graph.nodes()
    v_mag_pu_pfa = network.results.v_res(nodes=nodes, level=level)
    voltage = pd.DataFrame()
    for node in nodes:
        # check for over- and under-voltage
        try:
            voltage = pd.concat([voltage, v_mag_pu_pfa[repr(node)].rename(node.id)], axis=1)
        except KeyError:  # If not LV level of the node is available
            pass
    voltage.columns = [str(node) for node in voltage.columns]

    return voltage


def get_voltage_allowed_mv(network):
    """Source: mv_voltage_deviation(network, voltage_levels='mv_lv')"""

    v_dev_allowed_per_case = {'feedin_case_lower': 0.9,
                              'load_case_upper': 1.1}
    offset = network.config[
        'grid_expansion_allowed_voltage_deviations']['hv_mv_trafo_offset']
    control_deviation = network.config[
        'grid_expansion_allowed_voltage_deviations']['hv_mv_trafo_control_deviation']
    v_dev_allowed_per_case['feedin_case_upper'] = \
        1 + offset + control_deviation + network.config[
            'grid_expansion_allowed_voltage_deviations']['mv_feedin_case_max_v_deviation']
    v_dev_allowed_per_case['load_case_lower'] = \
        1 + offset - control_deviation - network.config[
            'grid_expansion_allowed_voltage_deviations']['mv_load_case_max_v_deviation']

    # maximum allowed apparent power of station in each time step
    v_dev_allowed_upper = \
        network.timeseries.timesteps_load_feedin_case.case.apply(
            lambda _: v_dev_allowed_per_case['{}_upper'.format(_)])
    v_dev_allowed_lower = \
        network.timeseries.timesteps_load_feedin_case.case.apply(
            lambda _: v_dev_allowed_per_case['{}_lower'.format(_)])

    return v_dev_allowed_lower, v_dev_allowed_upper


def get_voltage_allowed_lv(network):
    """Source: lv_voltage_deviation(network, mode=None, voltage_levels='mv_lv', timeindex=None)"""
    v_dev_allowed_per_case = {}
    v_allowed_lower, v_allowed_upper = pd.DataFrame(), pd.DataFrame()

    for lv_grid in network.mv_grid.lv_grids:
        # get voltage at primary side to calculate upper bound for
        # feed-in case and lower bound for load case
        v_lv_station_primary = network.results.v_res(nodes=[lv_grid.station], level='mv').iloc[:, 0]
        v_dev_allowed_per_case['feedin_case_upper'] = \
            v_lv_station_primary + network.config[
                'grid_expansion_allowed_voltage_deviations']['mv_lv_station_feedin_case_max_v_deviation']
        v_dev_allowed_per_case['load_case_lower'] = \
            v_lv_station_primary - network.config[
                'grid_expansion_allowed_voltage_deviations']['mv_lv_station_load_case_max_v_deviation']

        v_dev_allowed_per_case['feedin_case_lower'] = pd.Series(0.9, index=v_lv_station_primary.index)
        v_dev_allowed_per_case['load_case_upper'] = pd.Series(1.1, index=v_lv_station_primary.index)

        # maximum allowed voltage deviation in each time step
        v_allowed_upper[str(lv_grid.id)] = (
                ((network.timeseries.timesteps_load_feedin_case.case == 'load_case') *
                 v_dev_allowed_per_case['load_case_upper']) +
                ((network.timeseries.timesteps_load_feedin_case.case == 'feedin_case') *
                 v_dev_allowed_per_case['feedin_case_upper']))
        v_allowed_lower[str(lv_grid.id)] = (
                ((network.timeseries.timesteps_load_feedin_case.case == 'load_case') *
                 v_dev_allowed_per_case['load_case_lower']) +
                ((network.timeseries.timesteps_load_feedin_case.case == 'feedin_case') *
                 v_dev_allowed_per_case['feedin_case_lower']))

    return v_allowed_lower.dropna(), v_allowed_upper.dropna()


def get_voltage_overloads(network, return_allowed=False, return_is=False, level='mv'):
    """ Extracting over and undervoltages from edisgo object. Returns both as DataFrames with Index = Timesteps
    and Columns = MV Nodes
    Sources:
    - edisgo.flex_opt.check_tech_constraints.mv_voltage_deviation(network, voltage_levels='mv_lv')[source]
    - edisgo.flex_opt.check_tech_constraints.check_ten_percent_voltage_deviation(network)[source]"""

    v_dev_allowed_per_case = {'feedin_case_lower': 0.9, 'load_case_upper': 1.1}

    offset = network.config['grid_expansion_allowed_voltage_deviations']['hv_mv_trafo_offset']
    control_deviation = network.config['grid_expansion_allowed_voltage_deviations'][
        'hv_mv_trafo_control_deviation']
    v_dev_allowed_per_case['feedin_case_upper'] = 1 + offset + control_deviation + network.config[
        'grid_expansion_allowed_voltage_deviations']['mv_lv_feedin_case_max_v_deviation']
    v_dev_allowed_per_case['load_case_lower'] = 1 + offset - control_deviation - network.config[
        'grid_expansion_allowed_voltage_deviations']['mv_lv_load_case_max_v_deviation']

    # maximum allowed apparent power of station in each time step
    v_dev_allowed_upper = network.timeseries.timesteps_load_feedin_case.case.apply(
        lambda _: v_dev_allowed_per_case['{}_upper'.format(_)])
    v_dev_allowed_lower = network.timeseries.timesteps_load_feedin_case.case.apply(
        lambda _: v_dev_allowed_per_case['{}_lower'.format(_)])
    if return_allowed:
        return v_dev_allowed_upper, v_dev_allowed_lower

    nodes = network.mv_grid.graph.nodes()
    v_mag_pu_pfa = network.results.v_res(nodes=nodes, level=level)
    overvoltage, undervoltage = pd.DataFrame(), pd.DataFrame()
    for node in nodes:
        # check for over- and under-voltage
        try:
            if return_is:
                overvoltage_n = (v_mag_pu_pfa[repr(node)] - v_dev_allowed_upper.loc[v_mag_pu_pfa.index]
                                 ).rename(node.id)
                undervoltage_n = (v_dev_allowed_lower.loc[v_mag_pu_pfa.index] - v_mag_pu_pfa[repr(node)]
                                  ).rename(node.id)
            else:
                overvoltage_n = (v_mag_pu_pfa[repr(node)] - v_dev_allowed_upper.loc[v_mag_pu_pfa.index]
                                 ).div(v_dev_allowed_upper.loc[v_mag_pu_pfa.index]).rename(node.id)
                undervoltage_n = (v_dev_allowed_lower.loc[v_mag_pu_pfa.index] - v_mag_pu_pfa[repr(node)]
                                  ).div(v_dev_allowed_lower.loc[v_mag_pu_pfa.index]).rename(node.id)
            overvoltage = pd.concat([overvoltage, overvoltage_n], axis=1)
            undervoltage = pd.concat([undervoltage, undervoltage_n], axis=1)
        except KeyError:  # If not LV level of the node is available
            pass

    overvoltage.columns = [str(node) for node in overvoltage.columns]
    undervoltage.columns = [str(node) for node in undervoltage.columns]

    return overvoltage, undervoltage


def lv_voltage_deviation(network, mode=None, voltage_levels='mv_lv', timeindex=None):
    """
    Checks for voltage stability issues in LV grids.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
    mode : None or String
        If None voltage at all nodes in LV grid is checked. If mode is set to
        'stations' only voltage at busbar is checked.
    voltage_levels : :obj:`str`
        Specifies which allowed voltage deviations to use. Possible options
        are:

        * 'mv_lv'
          This is the default. The allowed voltage deviation for nodes in the
          MV grid is the same as for nodes in the LV grid. Further load and
          feed-in case are not distinguished.
        * 'lv'
          Use this to handle allowed voltage deviations in the MV and LV grid
          differently. Here, load and feed-in case are differentiated as well.
    timeindex : timesteps that were selected

    Returns
    -------
    :obj:`dict`
        Dictionary with :class:`~.grid.grids.LVGrid` as key and a
        :pandas:`pandas.DataFrame<dataframe>` with its critical nodes, sorted
        descending by voltage deviation, as value.
        Index of the dataframe are all nodes (of type
        :class:`~.grid.components.Generator`, :class:`~.grid.components.Load`,
        etc.) with over-voltage issues. Columns are 'v_mag_pu' containing the
        maximum voltage deviation as float and 'time_index' containing the
        corresponding time step the over-voltage occured in as
        :pandas:`pandas.Timestamp<timestamp>`.

    Notes
    -----
    Over-voltage is determined based on allowed voltage deviations defined in
    the config file 'config_grid_expansion' in section
    'grid_expansion_allowed_voltage_deviations'.

    """

    crit_nodes = {}

    v_dev_allowed_per_case = {}
    if voltage_levels == 'mv_lv':
        offset = network.config[
            'grid_expansion_allowed_voltage_deviations']['hv_mv_trafo_offset']
        control_deviation = network.config[
            'grid_expansion_allowed_voltage_deviations']['hv_mv_trafo_control_deviation']
        v_dev_allowed_per_case['feedin_case_upper'] = \
            1 + offset + control_deviation + network.config[
                'grid_expansion_allowed_voltage_deviations']['mv_lv_feedin_case_max_v_deviation']
        v_dev_allowed_per_case['load_case_lower'] = \
            1 + offset - control_deviation - network.config[
                'grid_expansion_allowed_voltage_deviations']['mv_lv_load_case_max_v_deviation']

        v_dev_allowed_per_case['feedin_case_lower'] = 0.9
        v_dev_allowed_per_case['load_case_upper'] = 1.1

        v_dev_allowed_upper = network.timeseries.timesteps_load_feedin_case.case.apply(
            lambda _: v_dev_allowed_per_case['{}_upper'.format(_)])
        v_dev_allowed_lower = network.timeseries.timesteps_load_feedin_case.case.apply(
            lambda _: v_dev_allowed_per_case['{}_lower'.format(_)])
    elif voltage_levels == 'lv':
        pass
    else:
        raise ValueError(
            'Specified mode {} is not a valid option.'.format(voltage_levels))

    for lv_grid in network.mv_grid.lv_grids:

        if mode:
            if mode == 'stations':
                nodes = [lv_grid.station]
            else:
                raise ValueError(
                    "{} is not a valid option for input variable 'mode' in "
                    "function lv_voltage_deviation. Try 'stations' or "
                    "None".format(mode))
        else:
            nodes = lv_grid.graph.nodes()

        if voltage_levels == 'lv':
            if mode == 'stations':
                # get voltage at primary side to calculate upper bound for
                # feed-in case and lower bound for load case
                v_lv_station_primary = network.results.v_res(
                    nodes=[lv_grid.station], level='mv').iloc[:, 0]
                timeindex = v_lv_station_primary.index
                v_dev_allowed_per_case['feedin_case_upper'] = \
                    v_lv_station_primary + network.config[
                        'grid_expansion_allowed_voltage_deviations'][
                        'mv_lv_station_feedin_case_max_v_deviation']
                v_dev_allowed_per_case['load_case_lower'] = \
                    v_lv_station_primary - network.config[
                        'grid_expansion_allowed_voltage_deviations'][
                        'mv_lv_station_load_case_max_v_deviation']
            else:
                # get voltage at secondary side to calculate upper bound for
                # feed-in case and lower bound for load case
                v_lv_station_secondary = network.results.v_res(
                    nodes=[lv_grid.station], level='lv').iloc[:, 0]
                if timeindex is None:
                    timeindex = v_lv_station_secondary.index
                v_dev_allowed_per_case['feedin_case_upper'] = \
                    v_lv_station_secondary + network.config[
                        'grid_expansion_allowed_voltage_deviations'][
                        'lv_feedin_case_max_v_deviation']
                v_dev_allowed_per_case['load_case_lower'] = \
                    v_lv_station_secondary - network.config[
                        'grid_expansion_allowed_voltage_deviations'][
                        'lv_load_case_max_v_deviation']
            v_dev_allowed_per_case['feedin_case_lower'] = pd.Series(
                0.9, index=timeindex)
            v_dev_allowed_per_case['load_case_upper'] = pd.Series(
                1.1, index=timeindex)

            # maximum allowed voltage deviation in each time step
            v_dev_allowed_upper = (
                ((network.timeseries.timesteps_load_feedin_case.case == 'load_case') *
                 v_dev_allowed_per_case['load_case_upper']) +
                ((network.timeseries.timesteps_load_feedin_case.case == 'feedin_case') *
                 v_dev_allowed_per_case['feedin_case_upper']))
            v_dev_allowed_lower = (
                ((network.timeseries.timesteps_load_feedin_case.case == 'load_case') *
                 v_dev_allowed_per_case['load_case_lower']) +
                ((network.timeseries.timesteps_load_feedin_case.case == 'feedin_case') *
                 v_dev_allowed_per_case['feedin_case_lower']))

        crit_nodes_grid = checks._voltage_deviation(network, nodes, v_dev_allowed_upper, v_dev_allowed_lower,
                                                    voltage_level='lv')

        if not crit_nodes_grid.empty:
            crit_nodes[lv_grid] = crit_nodes_grid.sort_values(
                by=['v_mag_pu'], ascending=False)

    if crit_nodes:
        if mode == 'stations':
            logger.debug(
                '==> {} LV station(s) has/have voltage issues.'.format(
                    len(crit_nodes)))
        else:
            logger.debug(
                '==> {} LV grid(s) has/have voltage issues.'.format(
                    len(crit_nodes)))
    else:
        if mode == 'stations':
            logger.debug('==> No voltage issues in LV stations.')
        else:
            logger.debug('==> No voltage issues in LV grids.')

    return crit_nodes


def analyze(edisgo, mode=None, timesteps=None, linear=False):
    """Analyzes the grid by power flow analysis

    Analyze the grid for violations of hosting capacity. Means, perform a
    power flow analysis and obtain voltages at nodes (load, generator,
    stations/transformers and branch tees) and active/reactive power at
    lines.

    The power flow analysis can currently only be performed for both grid
    levels MV and LV. See ToDos section for more information.

    A static `non-linear power flow analysis is performed using PyPSA
    <https://www.pypsa.org/doc/power_flow.html#full-non-linear-power-flow>`_.
    The high-voltage to medium-voltage transformer are not included in the
    analysis. The slack bus is defined at secondary side of these
    transformers assuming an ideal tap changer. Hence, potential
    overloading of the transformers is not studied here.

    Parameters
    ----------
    edisgo : edisgo object
    mode : str
        Allows to toggle between power flow analysis (PFA) on the whole
        grid topology (MV + LV), only MV or only LV. Defaults to None which
        equals power flow analysis for MV + LV which is the only
        implemented option at the moment. See ToDos section for
        more information.
    timesteps : :pandas:`pandas.DatetimeIndex<datetimeindex>` or \
        :pandas:`pandas.Timestamp<timestamp>`
        Timesteps specifies for which time steps to conduct the power flow
        analysis. It defaults to None in which case the time steps in
        timeseries.timeindex (see :class:`~.grid.network.TimeSeries`) are
        used.
    linear : boolean if linear pf should be used

    Notes
    -----
    The current implementation always translates the grid topology
    representation to the PyPSA format and stores it to
    :attr:`self.network.pypsa`.

    ToDos
    ------
    The option to export only the edisgo MV grid (mode = 'mv') to conduct
    a power flow analysis is implemented in
    :func:`~.tools.pypsa_io.to_pypsa` but NotImplementedError is raised
    since the rest of edisgo does not handle this option yet. The analyze
    function will throw an error since
    :func:`~.tools.pypsa_io.process_pfa_results`
    does not handle aggregated loads and generators in the LV grids. Also,
    grid reinforcement, pypsa update of time series, and probably other
    functionalities do not work when only the MV grid is analysed.

    Further ToDos are:
    * explain how power plants are modeled, if possible use a link
    * explain where to find and adjust power flow analysis defining
    parameters

    See Also
    --------
    :func:`~.tools.pypsa_io.to_pypsa`
        Translator to PyPSA data format

        """

    if timesteps is None:
        timesteps = edisgo.network.timeseries.timeindex
    # check if timesteps is array-like, otherwise convert to list
    if not hasattr(timesteps, "__len__"):
        timesteps = [timesteps]

    if edisgo.network.pypsa is None:
        # Translate eDisGo grid topology representation to PyPSA format
        edisgo.network.pypsa = pypsa_io.to_pypsa(
            edisgo.network, mode, timesteps)
    else:
        if edisgo.network.pypsa.edisgo_mode != mode:  # ORIGINAL: edisgo.network.pypsa.edisgo_mode is not mode
            # Translate eDisGo grid topology representation to PyPSA format
            edisgo.network.pypsa = pypsa_io.to_pypsa(
                edisgo.network, mode, timesteps)

    # check if all timesteps are in pypsa.snapshots, if not update time
    # series
    if False in [True if _ in edisgo.network.pypsa.snapshots else False for _ in timesteps]:
        pypsa_io.update_pypsa_timeseries(edisgo.network, timesteps=timesteps)

    # run power flow analysis
    if linear:
        edisgo.network.pypsa.lpf(timesteps)
        pypsa_io.process_pfa_results(edisgo.network,
                                     edisgo.network.pypsa,
                                     timesteps)
    else:
        pf_results = edisgo.network.pypsa.pf(timesteps)
        if not all(pf_results['converged']['0'].tolist()):
            failed_timesteps = pf_results['converged']['0'][~pf_results['converged']['0']].index
            logger.warning(
                '\n==================================================================================='
                '\nCould not converge on {} timesteps: \n{}'
                '\n==================================================================================='
                ''.format(failed_timesteps.shape[0], failed_timesteps))
            timesteps = timesteps.drop(failed_timesteps.tolist())

            # Drop failed timesteps from results
            pf_results['n_iter'] = pf_results['n_iter'].loc[timesteps]
            pf_results['error'] = pf_results['error'].loc[timesteps]
            pf_results['converged'] = pf_results['converged'].loc[timesteps]

        if all(pf_results['converged']['0'].tolist()):
            pypsa_io.process_pfa_results(edisgo.network,
                                         edisgo.network.pypsa,
                                         timesteps)
        else:
            raise ValueError("Power flow analysis did not converge.")

    return edisgo, timesteps


def reinforce(edisgo, inputs, **kwargs):
    """
    Reinforces the grid and calculates grid expansion costs.

    See :meth:`edisgo.flex_opt.reinforce_grid` for more information.

    """
    results = reinforce_grid(inputs,
                             edisgo=edisgo,
                             max_while_iterations=kwargs.get('max_while_iterations', 10),
                             copy_graph=kwargs.get('copy_graph', False),
                             timesteps_pfa=kwargs.get('timesteps_pfa', None),
                             combined_analysis=kwargs.get('combined_analysis', False),
                             mode=kwargs.get('mode', None),
                             linear=kwargs.get('linear', None))

    # add measure to Results object
    if not kwargs.get('copy_graph', False):
        edisgo.network.results.measures = 'grid_expansion'

    return results


def reinforce_grid(inputs, grid=None, edisgo=None, timesteps_pfa=None, copy_graph=False,
                   max_while_iterations=10, combined_analysis=False,
                   mode=None, linear=False, without_generator_import=True):
    """
    Evaluates grid reinforcement needs and performs measures.
    """

    def _add_lines_changes_to_equipment_changes():
        equipment, index, quantity = [], [], []
        for line, number_of_lines in lines_changes.items():
            equipment.append(line.type.name)
            index.append(line)
            quantity.append(number_of_lines)
        edisgo_reinforce.network.results.equipment_changes = \
            edisgo_reinforce.network.results.equipment_changes.append(
                pd.DataFrame(
                    {'iteration_step': [iteration_step] * len(
                        lines_changes),
                     'change': ['changed'] * len(lines_changes),
                     'equipment': equipment,
                     'quantity': quantity},
                    index=index))

    def _add_transformer_changes_to_equipment_changes(mode):
        for station, transformer_list in transformer_changes[mode].items():
            edisgo_reinforce.network.results.equipment_changes = \
                edisgo_reinforce.network.results.equipment_changes.append(
                    pd.DataFrame(
                        {'iteration_step': [iteration_step] * len(
                            transformer_list),
                         'change': [mode] * len(transformer_list),
                         'equipment': transformer_list,
                         'quantity': [1] * len(transformer_list)},
                        index=[station] * len(transformer_list)))

    timesteps_final = None
    if edisgo is None:
        edisgo = grid.edisgo

    # check if provided mode is valid
    if mode and mode is not 'mv':
        raise ValueError("Provided mode {} is not a valid mode.")

    # assign MV feeder to every generator, LV station, load, and branch tee
    # to assign grid expansion costs to an MV feeder
    assign_mv_feeder_to_nodes(edisgo.network.mv_grid)

    # analyze for all time steps (advantage is that load and feed-in case can
    # be obtained more performant in case `timesteps_pfa` = 'snapshot_analysis'
    # plus edisgo and edisgo_reinforce will have pypsa representation in case
    # reinforcement needs to be conducted on a copied graph)
    # edisgo = analyze_grid(edisgo, mode=mode, timesteps=timesteps_pfa, linear=linear), Outcommented by me = Problem?

    if timesteps_pfa is not None:
        # if timesteps_pfa = 'snapshot_analysis' get snapshots
        if isinstance(timesteps_pfa, str) and timesteps_pfa == 'snapshot_analysis':
            snapshots = tools.select_worstcase_snapshots(
                edisgo.network)
            # drop None values in case any of the two snapshots does not exist
            timesteps_pfa = pd.DatetimeIndex(data=[
                snapshots['load_case'], snapshots['feedin_case']]).dropna()
        # if timesteps_pfa is not of type datetime or does not contain
        # datetimes throw an error
        elif not isinstance(timesteps_pfa, datetime.datetime):
            if hasattr(timesteps_pfa, '__iter__'):
                if not all(isinstance(_, datetime.datetime)
                           for _ in timesteps_pfa):
                    raise ValueError(
                        'Input {} for timesteps_pfa is not valid.'.format(timesteps_pfa))
            else:
                raise ValueError(
                    'Input {} for timesteps_pfa is not valid.'.format(
                        timesteps_pfa))

    # Added by me to get pypsa object for original loads at initial run of grid
    if edisgo.network.pypsa is None:
        if timesteps_pfa is None:
            timesteps_pfa = edisgo.network.timeseries.timeindex
        # check if timesteps is array-like, otherwise convert to list
        if not hasattr(timesteps_pfa, "__len__"):
            timesteps_pfa = [timesteps_pfa]
        if edisgo.network.pypsa is None:
            # Translate eDisGo grid topology representation to PyPSA format
            edisgo.network.pypsa = pypsa_io.to_pypsa(
                edisgo.network, mode, timesteps_pfa)
        else:
            if edisgo.network.pypsa.edisgo_mode != mode:  # ORIGINAL: edisgo.network.pypsa.edisgo_mode is not mode
                # Translate eDisGo grid topology representation to PyPSA format
                edisgo.network.pypsa = pypsa_io.to_pypsa(
                    edisgo.network, mode, timesteps_pfa)

    iteration_step = 1
    # edisgo_reinforce.analyze(mode=mode, timesteps=timesteps_pfa)
    edisgo, timesteps_original = analyze(edisgo, mode=mode, timesteps=timesteps_pfa)

    # in case reinforcement needs to be conducted on a copied graph the
    # edisgo object is deep copied
    if copy_graph is True:
        edisgo_reinforce = copy.deepcopy(edisgo)
    else:
        edisgo_reinforce = edisgo

    problems_post = {}

    # REINFORCE OVERLOADED TRANSFORMERS AND LINES
    logger.debug('==> Check station load.')
    overloaded_mv_station = checks.hv_mv_station_load(edisgo_reinforce.network)
    problems_post['hv_mv_station_load'] = overloaded_mv_station
    overloaded_lv_stations = checks.mv_lv_station_load(
        edisgo_reinforce.network)
    problems_post['mv_lv_station_load'] = overloaded_lv_stations
    logger.debug('==> Check line load.')
    crit_lines = checks.mv_line_load(edisgo_reinforce.network)
    problems_post['mv_line_load'] = crit_lines
    if not mode:
        crit_lines = crit_lines.append(
            checks.lv_line_load(edisgo_reinforce.network))
        problems_post['lv_line_load'] = crit_lines
    if grid:
        grid.trafo_s_original = grid.get_aggregated_overloads(inputs, transformer=True, return_is=True,
                                                              network=edisgo_reinforce.network,
                                                              timesteps=timesteps_original)
        grid.line_s_original = grid.get_aggregated_overloads(inputs, line=True, return_is=True,
                                                             network=edisgo_reinforce.network,
                                                             timesteps=timesteps_original)

    problems_post['hv_mv_station_load2'] = []
    problems_post['mv_lv_station_load2'] = []
    problems_post['mv_line_load2'] = []
    problems_post['lv_line_load2'] = []
    while_counter = 0
    while ((not overloaded_mv_station.empty or not overloaded_lv_stations.empty or not crit_lines.empty) and
            while_counter < max_while_iterations):
        if not overloaded_mv_station.empty:
            # reinforce substations
            transformer_changes = \
                reinforce_measures.extend_substation_overloading(
                    edisgo_reinforce.network, overloaded_mv_station)
            # write added and removed transformers to results.equipment_changes
            _add_transformer_changes_to_equipment_changes('added')
            _add_transformer_changes_to_equipment_changes('removed')

        if not overloaded_lv_stations.empty:
            # reinforce distribution substations
            transformer_changes = \
                reinforce_measures.extend_distribution_substation_overloading(
                    edisgo_reinforce.network, overloaded_lv_stations)
            # write added and removed transformers to results.equipment_changes
            _add_transformer_changes_to_equipment_changes('added')
            _add_transformer_changes_to_equipment_changes('removed')

        if not crit_lines.empty:
            # reinforce lines
            lines_changes = reinforce_measures.reinforce_branches_overloading(
                edisgo_reinforce.network, crit_lines)
            # write changed lines to results.equipment_changes
            _add_lines_changes_to_equipment_changes()

        # run power flow analysis again (after updating pypsa object) and check
        # if all over-loading problems were solved
        logger.debug('==> Run power flow analysis.')
        pypsa_io.update_pypsa_grid_reinforcement(
            edisgo_reinforce.network,
            edisgo_reinforce.network.results.equipment_changes[
                edisgo_reinforce.network.results.equipment_changes.iteration_step == iteration_step])
        # edisgo_reinforce.analyze(mode=mode, timesteps=timesteps_pfa)
        edisgo_reinforce, timesteps_final = \
            analyze(edisgo_reinforce, mode=mode, linear=linear, timesteps=timesteps_pfa)

        logger.debug('==> Recheck station load.')
        overloaded_mv_station = checks.hv_mv_station_load(
            edisgo_reinforce.network)
        problems_post['hv_mv_station_load2'].append(overloaded_mv_station)
        overloaded_lv_stations = checks.mv_lv_station_load(
            edisgo_reinforce.network)
        problems_post['mv_lv_station_load2'].append(overloaded_lv_stations)
        logger.debug('==> Recheck line load.')
        crit_lines = checks.mv_line_load(edisgo_reinforce.network)
        problems_post['mv_line_load2'].append(crit_lines)
        if not mode:
            crit_lines = crit_lines.append(
                checks.lv_line_load(edisgo_reinforce.network))
            problems_post['lv_line_load2'].append(crit_lines)

        iteration_step += 1
        while_counter += 1

    # check if all load problems were solved after maximum number of
    # iterations allowed
    if (while_counter == max_while_iterations and
            (not crit_lines.empty or not overloaded_mv_station.empty or not overloaded_lv_stations.empty)):
        edisgo_reinforce.network.results.unresolved_issues.update(crit_lines)
        edisgo_reinforce.network.results.unresolved_issues.update(
            overloaded_lv_stations)
        edisgo_reinforce.network.results.unresolved_issues.update(
            overloaded_mv_station)
        raise exceptions.MaximumIterationError(
            "Overloading issues for the following lines, lv- and mv-stations could not be solved:"
            "\n{}\n{}\n{}".format(crit_lines, overloaded_lv_stations, overloaded_mv_station))
    else:
        logger.info('==> Load issues were solved in {} iteration '
                    'step(s).'.format(while_counter))

    # REINFORCE BRANCHES DUE TO VOLTAGE ISSUES
    iteration_step += 1

    # solve voltage problems in MV grid
    logger.debug('==> Check voltage in MV grid.')
    if combined_analysis:
        voltage_levels = 'mv_lv'
    else:
        voltage_levels = 'mv'
    crit_nodes = checks.mv_voltage_deviation(edisgo_reinforce.network,
                                             voltage_levels=voltage_levels)
    if grid:
        # Potentially rename over_v_original -> mv_side, under_v_original -> lv_side
        grid.over_v_original, grid.under_v_original = {'min': None, 'max': None, 'is': None}, \
                                                      {'min': None, 'max': None, 'is': None}
        grid.over_v_original['min'], grid.over_v_original['max'] = \
            grid.get_aggregated_overloads(inputs,
                                          voltage=True,
                                          return_allowed=True,
                                          network=edisgo_reinforce.network,
                                          timesteps=timesteps_original,
                                          level='mv')
        grid.over_v_original['is'] = \
            grid.get_aggregated_overloads(inputs,
                                          voltage=True,
                                          return_is=True,
                                          network=edisgo_reinforce.network,
                                          timesteps=timesteps_original,
                                          level='mv')
    problems_post['mv_voltage_deviation'] = crit_nodes
    problems_post['mv_voltage_deviation2'] = []
    while_counter = 0
    while crit_nodes and while_counter < max_while_iterations:

        # reinforce lines
        lines_changes = reinforce_measures.reinforce_branches_overvoltage(
            edisgo_reinforce.network, edisgo_reinforce.network.mv_grid,
            crit_nodes[edisgo_reinforce.network.mv_grid])
        # write changed lines to results.equipment_changes
        _add_lines_changes_to_equipment_changes()

        # run power flow analysis again (after updating pypsa object) and check
        # if all over-voltage problems were solved
        logger.debug('==> Run power flow analysis.')
        pypsa_io.update_pypsa_grid_reinforcement(
            edisgo_reinforce.network,
            edisgo_reinforce.network.results.equipment_changes[
                edisgo_reinforce.network.results.equipment_changes.iteration_step == iteration_step])
        # edisgo_reinforce.analyze(mode=mode, timesteps=timesteps_pfa)
        edisgo_reinforce, timesteps_final = \
            analyze(edisgo_reinforce, mode=mode, linear=linear, timesteps=timesteps_pfa)

        logger.debug('==> Recheck voltage in MV grid.')
        crit_nodes = checks.mv_voltage_deviation(edisgo_reinforce.network,
                                                 voltage_levels=voltage_levels)
        problems_post['mv_voltage_deviation2'].append(crit_nodes)
        iteration_step += 1
        while_counter += 1

    # check if all voltage problems were solved after maximum number of
    # iterations allowed
    if while_counter == max_while_iterations and crit_nodes:
        for k, v in crit_nodes.items():
            for node in v.index:
                edisgo_reinforce.network.results.unresolved_issues.update(
                    {repr(node): v.loc[node, 'v_mag_pu']})
        raise exceptions.MaximumIterationError(
            "Over-voltage issues for the following nodes in MV grid could "
            "not be solved: {}".format(crit_nodes))
    else:
        logger.info('==> Voltage issues in MV grid were solved in {} '
                    'iteration step(s).'.format(while_counter))

    # solve voltage problems at secondary side of LV stations
    logger.debug('==> Check voltage at secondary side of LV stations.')
    if combined_analysis:
        voltage_levels = 'mv_lv'
    else:
        voltage_levels = 'lv'
    crit_stations = lv_voltage_deviation(edisgo_reinforce.network,
                                         mode='stations',
                                         voltage_levels=voltage_levels)  # original: checks.lv_voltage_deviation()

    problems_post['lv_voltage_deviation'] = crit_stations
    if grid:
        grid.under_v_original['min'], grid.under_v_original['max'] = \
            grid.get_aggregated_overloads(inputs,
                                          voltage=True,
                                          return_allowed=True,
                                          network=edisgo_reinforce.network,
                                          timesteps=timesteps_original,
                                          level='lv')
        grid.under_v_original['is'] = \
            grid.get_aggregated_overloads(inputs,
                                          voltage=True,
                                          return_is=True,
                                          network=edisgo_reinforce.network,
                                          timesteps=timesteps_original,
                                          level='lv')

    problems_post['lv_voltage_deviation2'] = []
    while_counter = 0
    while crit_stations and while_counter < max_while_iterations:
        # reinforce distribution substations
        transformer_changes = \
            reinforce_measures.extend_distribution_substation_overvoltage(
                edisgo_reinforce.network, crit_stations)
        # write added transformers to results.equipment_changes
        _add_transformer_changes_to_equipment_changes('added')

        # run power flow analysis again (after updating pypsa object) and check
        # if all over-voltage problems were solved
        logger.debug('==> Run power flow analysis.')
        pypsa_io.update_pypsa_grid_reinforcement(
            edisgo_reinforce.network,
            edisgo_reinforce.network.results.equipment_changes[
                edisgo_reinforce.network.results.equipment_changes.iteration_step == iteration_step])
        # edisgo_reinforce.analyze(mode=mode, timesteps=timesteps_pfa)
        edisgo_reinforce, timesteps_final = \
            analyze(edisgo_reinforce, mode=mode, linear=linear, timesteps=timesteps_pfa)

        logger.debug('==> Recheck voltage at secondary side of LV stations.')
        crit_stations = checks.lv_voltage_deviation(
            edisgo_reinforce.network, mode='stations',
            voltage_levels=voltage_levels)
        problems_post['lv_voltage_deviation2'].append(crit_stations)

        iteration_step += 1
        while_counter += 1

    # check if all voltage problems were solved after maximum number of
    # iterations allowed
    if while_counter == max_while_iterations and crit_stations:
        for k, v in crit_stations.items():
            for node in v.index:
                edisgo_reinforce.network.results.unresolved_issues.update(
                    {repr(node): v.loc[node, 'v_mag_pu']})
        raise exceptions.MaximumIterationError(
            "Over-voltage issues at busbar could not be solved for the "
            "following LV grids: {}".format(crit_stations))
    else:
        logger.info('==> Voltage issues at busbars in LV grids were solved '
                    'in {} iteration step(s).'.format(while_counter))

    # solve voltage problems in LV grids
    if not mode:
        logger.debug('==> Check voltage in LV grids.')
        crit_nodes = checks.lv_voltage_deviation(edisgo_reinforce.network,
                                                 voltage_levels=voltage_levels)
        problems_post['lv_voltage_deviation_nodes'] = crit_nodes

        problems_post['lv_voltage_deviation_nodes2'] = []
        while_counter = 0
        while crit_nodes and while_counter < max_while_iterations:
            # for every grid in crit_nodes do reinforcement
            for grid in crit_nodes:
                # reinforce lines
                lines_changes = \
                    reinforce_measures.reinforce_branches_overvoltage(
                        edisgo_reinforce.network, grid, crit_nodes[grid])
                # write changed lines to results.equipment_changes
                _add_lines_changes_to_equipment_changes()

            # run power flow analysis again (after updating pypsa object) and
            # check if all over-voltage problems were solved
            logger.debug('==> Run power flow analysis.')
            pypsa_io.update_pypsa_grid_reinforcement(
                edisgo_reinforce.network,
                edisgo_reinforce.network.results.equipment_changes[
                    edisgo_reinforce.network.results.equipment_changes.iteration_step == iteration_step])
            # edisgo_reinforce.analyze(mode=mode, timesteps=timesteps_pfa)
            edisgo_reinforce, timesteps_final = \
                analyze(edisgo_reinforce, mode=mode, linear=linear, timesteps=timesteps_pfa)

            logger.debug('==> Recheck voltage in LV grids.')
            crit_nodes = checks.lv_voltage_deviation(
                edisgo_reinforce.network, voltage_levels=voltage_levels)
            problems_post['lv_voltage_deviation_nodes2'].append(crit_nodes)

            iteration_step += 1
            while_counter += 1

        # check if all voltage problems were solved after maximum number of
        # iterations allowed
        if while_counter == max_while_iterations and crit_nodes:
            for k, v in crit_nodes.items():
                for node in v.index:
                    edisgo_reinforce.network.results.unresolved_issues.update(
                        {repr(node): v.loc[node, 'v_mag_pu']})
            raise exceptions.MaximumIterationError(
                "Over-voltage issues for the following nodes in LV grids "
                "could not be solved: {}".format(crit_nodes))
        else:
            logger.info(
                '==> Voltage issues in LV grids were solved '
                'in {} iteration step(s).'.format(while_counter))

    # RECHECK FOR OVERLOADED TRANSFORMERS AND LINES
    logger.debug('==> Recheck station load.')
    overloaded_mv_station = checks.hv_mv_station_load(edisgo_reinforce.network)
    problems_post['hv_mv_station_load_re'] = overloaded_mv_station
    overloaded_lv_stations = checks.mv_lv_station_load(
        edisgo_reinforce.network)
    problems_post['mv_lv_station_load_re'] = overloaded_lv_stations
    logger.debug('==> Recheck line load.')
    crit_lines = checks.mv_line_load(edisgo_reinforce.network)
    problems_post['mv_line_load_re'] = crit_lines
    if not mode:
        crit_lines = crit_lines.append(
            checks.lv_line_load(edisgo_reinforce.network))
        problems_post['lv_line_load_re'] = crit_lines

    problems_post['hv_mv_station_load_re2'] = []
    problems_post['mv_lv_station_load_re2'] = []
    problems_post['mv_line_load_re2'] = []
    problems_post['lv_line_load_re2'] = []
    while_counter = 0
    while ((not overloaded_mv_station.empty or not overloaded_lv_stations.empty
            or not crit_lines.empty) and while_counter < max_while_iterations):

        if not overloaded_mv_station.empty:
            # reinforce substations
            transformer_changes = \
                reinforce_measures.extend_substation_overloading(
                    edisgo_reinforce.network, overloaded_mv_station)
            # write added and removed transformers to results.equipment_changes
            _add_transformer_changes_to_equipment_changes('added')
            _add_transformer_changes_to_equipment_changes('removed')

        if not overloaded_lv_stations.empty:
            # reinforce substations
            transformer_changes = \
                reinforce_measures.extend_distribution_substation_overloading(
                    edisgo_reinforce.network, overloaded_lv_stations)
            # write added and removed transformers to results.equipment_changes
            _add_transformer_changes_to_equipment_changes('added')
            _add_transformer_changes_to_equipment_changes('removed')

        if not crit_lines.empty:
            # reinforce lines
            lines_changes = reinforce_measures.reinforce_branches_overloading(
                edisgo_reinforce.network, crit_lines)
            # write changed lines to results.equipment_changes
            _add_lines_changes_to_equipment_changes()

        # run power flow analysis again (after updating pypsa object) and check
        # if all over-loading problems were solved
        logger.debug('==> Run power flow analysis.')
        pypsa_io.update_pypsa_grid_reinforcement(
            edisgo_reinforce.network,
            edisgo_reinforce.network.results.equipment_changes[
                edisgo_reinforce.network.results.equipment_changes.iteration_step == iteration_step])
        # edisgo_reinforce.analyze(mode=mode, timesteps=timesteps_pfa)
        edisgo_reinforce, timesteps_final = \
            analyze(edisgo_reinforce, mode=mode, linear=linear, timesteps=timesteps_pfa)

        logger.debug('==> Recheck station load.')
        overloaded_mv_station = checks.hv_mv_station_load(
            edisgo_reinforce.network)
        problems_post['hv_mv_station_load_re2'].append(overloaded_mv_station)
        overloaded_lv_stations = checks.mv_lv_station_load(
            edisgo_reinforce.network)
        problems_post['mv_lv_station_load_re2'].append(overloaded_lv_stations)
        logger.debug('==> Recheck line load.')
        crit_lines = checks.mv_line_load(edisgo_reinforce.network)
        problems_post['mv_line_load_re2'].append(crit_lines)
        if not mode:
            crit_lines = crit_lines.append(
                checks.lv_line_load(edisgo_reinforce.network))
            problems_post['lv_line_load_re2'].append(crit_lines)

        iteration_step += 1
        while_counter += 1

    # check if all load problems were solved after maximum number of
    # iterations allowed
    if (while_counter == max_while_iterations and
            (not crit_lines.empty or not overloaded_mv_station.empty or not overloaded_lv_stations.empty)):
        edisgo_reinforce.network.results.unresolved_issues.update(crit_lines)
        edisgo_reinforce.network.results.unresolved_issues.update(
            overloaded_lv_stations)
        edisgo_reinforce.network.results.unresolved_issues.update(
            overloaded_mv_station)
        raise exceptions.MaximumIterationError(
            "Overloading issues (after solving over-voltage issues) for the"
            "following lines could not be solved: {}".format(crit_lines))
    else:
        logger.info(
            '==> Load issues were rechecked and solved '
            'in {} iteration step(s).'.format(while_counter))

    # final check 10% criteria
    checks.check_ten_percent_voltage_deviation(edisgo_reinforce.network)

    # calculate grid expansion costs
    # If-clause added by me due to AttributeError: 'DataFrame' object has no attribute 'iteration_step' in
    # network.results.equipment_changes.iteration_step > 0]
    if not edisgo_reinforce.network.results.equipment_changes.empty:
        edisgo_reinforce.network.results.grid_expansion_costs = \
            grid_expansion_costs(edisgo_reinforce.network, without_generator_import=without_generator_import)
    else:
        edisgo_reinforce.network.results.grid_expansion_costs = pd.DataFrame(
            {'type': ['N/A'],
             'total_costs': [0],
             'length': [0],
             'quantity': [0],
             'voltage_level': '',
             'mv_feeder': ''
             },
            index=['No reinforced equipment.'])

    if grid:
        grid.edisgo = edisgo_reinforce
        grid.problems_post = problems_post
        grid.timesteps['original'] = timesteps_original
        grid.timesteps['final'] = timesteps_final if timesteps_final is not None else timesteps_original

        return grid
    else:
        return edisgo_reinforce
