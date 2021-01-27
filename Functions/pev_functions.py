import copy
import pandas as pd
import numpy as np
from random import choices
from Functions import config_functions as cfg
import logging

logger = logging.getLogger(__name__)
cfg.set_logging()


class PEVInfo:
    __slots__ = ('charging_inputs', 'data', 'ppev_load', 'connected', 'overnight', 'soc', 'soc_need', 'remaining_t',
                 'charge_load', 'original_load_delta', 'remaining_load_delta', 'charging_history',
                 'fleet_cc_capa', 'fleet_dc_capa', 'rides_at_node', 'ppev_load_must', 'ppev_load_smart',
                 'ppev_load_avrg', 'pev_select', 'charger_attr', 'number_chargers', 'node_batt_capa', 'smart_charger')

    def __init__(self, rawdata, inputs):
        """Set trips and technical data of all PEV at the node
        self.data variables:
        # [0] = ride ID, [1] = ride in a day ID, [2] = arr_t, [3] = batt_capacity, [4] = demand_type, [5] = dep_t,
        # [6] = energy_start, [7] = energy_tb_charged [kWh], [8] = node, [9] = energy_end, [10] = soc,
        # [11] = energy [kWh], [12] = charge_load [kW], [13] = day number, [14] = charged [kWh], [15] = discharged [kWh]
        """
        self.data = rawdata
        self.charging_history = None  # [kW]
        self.ppev_load = None  # [kW], doesn't include all PEV charging in output because of only_represented_days
        self.ppev_load_must = None  # [kW], doesn't include all PEV charging in output because of only_represented_days
        self.ppev_load_smart = None
        self.ppev_load_avrg = None
        self.rides_at_node = None  # [kW]
        self.fleet_cc_capa = None  # [kW]
        self.fleet_dc_capa = None  # [kW]
        self.original_load_delta = None  # [kW]
        self.remaining_load_delta = None  # [kW]
        self.charging_inputs = [inputs.p_charge, cfg.t_periods, cfg.soc_min]
        self.pev_select = []
        self.charger_attr = {'smart_cc_share': 0.0,
                             'charger_availability': 0.0}
        self.number_chargers = {'total': 0,
                                'total_smart': 0,
                                'max_rides': np.array([]),
                                'chargers': np.array([]),
                                'smart_chargers': np.array([])}
        self.node_batt_capa = np.array([])
        self.smart_charger = np.array([])

    def get_pev_select(self, inputs, grid, t):
        a = int(max(0,
                    ((self.data.shape[0] - 1) / inputs.index.size) * t - grid.pkw_number * 3))
        b = int(min((self.data.shape[0] - 1),
                    ((self.data.shape[0] - 1) / inputs.index.size) * t + grid.pkw_number * 3))
        times = self.data[['arr_t', 'dep_t']][a:b]
        pev_select = np.array(range(a, b))[(times['arr_t'] <= t) *
                                           (times['dep_t'] > t)]

        return pev_select

    def get_number_chargers(self, smart_cc_share, charger_availability, inputs, grid, first=False):
        if (smart_cc_share != self.charger_attr['smart_cc_share'] or
                cfg.charger_availability != self.charger_attr['charger_availability'] or
                self.number_chargers['total_smart'] == 0):

            self.charger_attr['charger_availability'] = charger_availability

            if smart_cc_share != self.charger_attr['smart_cc_share'] and not first:
                self.charger_attr['smart_cc_share'] = smart_cc_share
                self.number_chargers['smart_chargers'] = (
                        self.number_chargers['chargers'] * self.charger_attr['smart_cc_share']).round(0).astype(int)
                self.number_chargers['total_smart'] = int(self.number_chargers['smart_chargers'].sum())
            else:
                self.charger_attr['smart_cc_share'] = smart_cc_share

                if self.rides_at_node is None:
                    self.rides_at_node = np.zeros([inputs.index.size, len(grid.nodes['node'])])
                    for t in range(inputs.index.size):  # [2] = arrival time, [5] = departure time
                        pev_select = self.get_pev_select(inputs, grid, t)
                        try:
                            ridesatnode = np.unique(self.data['node'][pev_select], return_counts=True)
                            self.rides_at_node[t, ridesatnode[0]] = ridesatnode[1]
                        except IndexError:
                            logger.error('IndexError: arrays used as indices must be of integer (or boolean) type')
                self.rides_at_node = self.rides_at_node.astype('int')

                # Get the number of charging ports that need to be available at a given node
                self.number_chargers['chargers'] = np.zeros(self.rides_at_node.shape[1], dtype=int)
                self.number_chargers['max_rides'] = self.rides_at_node.max(0)
                for node in range(self.rides_at_node.shape[1]):
                    unique, counts = np.unique(self.rides_at_node[:, node], return_counts=True)
                    self.number_chargers['chargers'][node] = unique[np.where(
                        counts.cumsum() >= self.rides_at_node.shape[0] *
                        self.charger_attr['charger_availability'])[0]][0]
                self.number_chargers['smart_chargers'] = (
                            self.number_chargers['chargers'] * self.charger_attr['smart_cc_share']).round(0).astype(int)
                self.number_chargers['total'] = int(self.number_chargers['chargers'].sum())
                self.number_chargers['total_smart'] = int(self.number_chargers['smart_chargers'].sum())

    def all_to_dataframe(self, inputs, node_names):
        def as_dataframe(data):
            index = inputs.dates_represented_index if inputs.downsample else inputs.index
            try:
                out = pd.DataFrame(data, index=index, columns=node_names)
            except ValueError:
                out = pd.DataFrame(data[0, :, :], index=index, columns=node_names)
            return out

        self.fleet_cc_capa = as_dataframe(self.fleet_cc_capa)
        self.fleet_dc_capa = as_dataframe(self.fleet_dc_capa)
        self.rides_at_node = as_dataframe(self.rides_at_node)
        self.ppev_load = as_dataframe(self.ppev_load)
        self.ppev_load_must = as_dataframe(self.ppev_load_must)
        self.original_load_delta = as_dataframe(self.original_load_delta)  # [kW]
        self.remaining_load_delta = as_dataframe(self.remaining_load_delta)  # [kW]

    def add_charging(self, t, remaining_pev, must=False):
        """
        For-loop: Aggregated charging of PEV to the node level to save current charge load of a PEV
        remaining_pev[0] = charge load
        remaining_pev[1] = node number
        remaining_pev[2] = pev number

        if must: Charging PEV that must charge -> Add losses on grid side, PEV load is fix
        if not must:  # Charging grid-beneficial -> Subtract charging losses on PEV side, Grid load is fix

        pevs.ppev_load *= ((pevs.ppev_load > 0) / cfg.efficiency +
                           (pevs.ppev_load < 0) * cfg.efficiency)  # Add losses from dis-/charging
        """

        for node in set(remaining_pev[1]):
            charging = remaining_pev[2, np.where(remaining_pev[1] == node)].sum().astype('float32')  # [kW]
            if must:
                charging *= ((charging > 0) / cfg.efficiency +
                             (charging < 0) * cfg.efficiency)  # Add losses from dis-/charging
            self.ppev_load[t, int(node)] += charging
            if not must:
                self.remaining_load_delta[t, int(node)] -= charging  # [kW], Subtract charging of PEV

        remaining_pevs = remaining_pev[0].astype(int)
        remaining_pevs_load = remaining_pev[2]
        if not must:
            remaining_pevs_load *= ((remaining_pevs_load > 0) * cfg.efficiency +
                                    (remaining_pevs_load < 0) / cfg.efficiency)  # Add losses from dis-/charging
        self.data['charge_load'][remaining_pevs] += remaining_pevs_load  # [kW]
        self.data['charged'][remaining_pevs] += np.clip(remaining_pevs_load * (24 / cfg.t_periods),  # [kW] -> [kWh]
                                                        a_min=0, a_max=None)
        self.data['discharged'][remaining_pevs] += np.clip(remaining_pevs_load * (24 / cfg.t_periods),  # [kW]->[kWh]
                                                           a_min=None, a_max=0)

        # KEEP!!! see v2g_cost_losses_battery:
        if self.charging_history is not None:
            self.charging_history[remaining_pevs, t % 48] += remaining_pevs_load  # [kW]

    def targeted_charging(self, ride_data, rides, t, target, discharge=False, new=False, avrg_factor=1.0):
        """ Charge up to a certain target if possible (overloads if smart charging and free capacity if
        average charging"""

        if rides['pevs'].size != 0:  # Only charge if PEV are available
            # Fully charge all PEV until load_delta is reached
            ride_data, rides = self.cc_per_node(t, rides, ride_data, target,
                                                cumulative=True, discharge=discharge, new=new)
            if rides['pevs'].size != 0:  # Continue if PEV are remaining
                # Charge remaining load_delta from individual PEV
                ride_data, rides = self.cc_per_node(t, rides, ride_data, target, discharge=discharge, new=new)

        return ride_data, rides

    def cc_per_node(self, t, rides, ride_data, target_level, discharge=False, cumulative=False, output=True, new=False):
        """ Set up a matrix that defines which PEV should be charging to fulfill the load_delta:
          - Nodes[0] = CC_score for each of the PEV to sort for the charging order
          - Nodes[1] = Maximum load that a PEV can charge/discharge
          - Nodes[2] = Require load_delta at the node of the PEV
          - Nodes[3] = Cumulative charging load that can be achieved per node
          - Nodes[4] = Number of PEVs at a node
          - Nodes[5] = PEV number
          - Nodes[6] = Node number

          - Charge_node_bool[0] = Boolean if the cumulative charge load is larger than the load_delta
          - Charge_node_bool[1] = Boolean - maximum charge load of the first PEV at a node is larger than cc_max"""

        nodes = np.stack([ride_data['cc_score'],
                          ride_data['dc_max'] if discharge else ride_data['cc_max'],
                          target_level[rides['nodes']],
                          np.zeros(len(rides['pevs'])),
                          np.ones(len(rides['pevs'])),
                          rides['pevs'],
                          rides['nodes']])
        nodes = nodes[:, (nodes[1] < 0) if discharge else (nodes[1] > 0)]  # Drop out PEV that can not charge/discharge
        nodes = nodes[:, np.argsort(nodes[0])[::1 if discharge else -1]]  # Sort by cc_score, inverse for discharging
        try:
            cumsum_by_nodes = quick_groupby(nodes[6], nodes[[1, 4]].transpose(), np.cumsum)
            nodes[3] = cumsum_by_nodes[:, 0]
            nodes[4] = cumsum_by_nodes[:, 1]
        except ValueError:  # None of the PEV actually have capacity to charge
            return ride_data, rides

        if cumulative:  # cumulative charge_max >= load_delta?
            pev_select_f = (nodes[3] >= nodes[2]) * (nodes[2] <= 0) * (nodes[1] != 0) if discharge else \
                (nodes[3] <= nodes[2]) * (nodes[2] >= 0) * (nodes[1] != 0)
        else:  # charge_max <= load_delta?
            pev_select_f = \
                ((nodes[1] <= nodes[2]) * (nodes[4] == 1) * (nodes[2] <= 0) * (nodes[1] != 0)) if discharge else \
                ((nodes[1] >= nodes[2]) * (nodes[4] == 1) * (nodes[2] >= 0) * (nodes[1] != 0))

        # Selection to cumulatively fulfill load_delta:
        charging = np.stack([nodes[5][pev_select_f],  # [5] = pev number
                             nodes[6][pev_select_f],  # [6] = node number
                             (nodes[1] if cumulative else  # [1] = charging power [kW]
                              nodes[2])[pev_select_f]]  # [2] = load_delta [kW]
                            ).astype('float32')

        if charging[0].size != 0:
            self.add_charging(t, charging)
            pev_not_charging = ~np.in1d(rides['pevs'], charging[0])  # Element of charging[0] in rides['pevs'] ?
            if not new:
                ride_data = ride_data[pev_not_charging]  # Remove PEV that are charging already
                rides = rides[pev_not_charging]  # Remove PEV that are charging already
        if output:
            return ride_data, rides

    @staticmethod
    def get_node_allowed(grid, load_potential, t, cases, discharging=False, charging=False, free=False):

        # Change feedin-case to load case if curtailment or PEV charging has the potential to change the case
        if cases[1] == 'feedin_case':
            transformer_sallowed = grid.transformer_sallowed * cfg.lv_feedin_case_transformer
            line_sallowed = grid.line_sallowed * cfg.mv_feedin_case_line

            # Indicates trafo and line problems in current grid
            line_problem = grid.lines_to_node(np.ones(len(grid.nodes['node'])),
                                              abs(grid.line_s[t]) > line_sallowed) > 0
            trafo_problem = abs(grid.transformer_s[t]) > transformer_sallowed
            any_problem = line_problem | trafo_problem

            # residual_load in kW
            if (cases[0] + load_potential[any_problem].sum()) > 0:  # kW
                transformer_sallowed = grid.transformer_sallowed * cfg.lv_load_case_transformer
                line_sallowed = grid.line_sallowed * cfg.mv_load_case_line
        else:
            transformer_sallowed = grid.transformer_sallowed * cfg.lv_load_case_transformer
            line_sallowed = grid.line_sallowed * cfg.mv_load_case_line

        if free:  # E_free_r, E_free_l: Free capacity at the transformer, line
            # Differentiate RES and Demand - if RES, a lot more can be charged!
            # Free charging capacity: > 0 = charging possible, < 0 = discharging possible
            if discharging:
                # Capacity minus RES can be discharged if RES case
                # Capacity plus DEMAND can be discharged if DEMAND case
                E_r = np.minimum(-(transformer_sallowed + grid.transformer_s[t]), 0)  # [kW]
                E_l = np.minimum(grid.lines_to_node(load_potential, -(line_sallowed + grid.line_s[t])), 0)

                # Return minimum free capacity at node
                return np.maximum(E_r, E_l)
            elif charging:
                # Capacity plus RES can be charged if RES case
                # Capacity minus DEMAND can be charged if DEMAND case
                E_r = np.maximum(transformer_sallowed - grid.transformer_s[t], 0)  # [kW]
                E_l = np.maximum(grid.lines_to_node(load_potential, (line_sallowed - grid.line_s[t])), 0)

                # Return minimum free capacity at node
                return np.minimum(E_r, E_l)
        else:
            # E_plus_r, E_plus_l: Problems at the transformer, line from loads
            E_plus_r = -np.minimum(transformer_sallowed - grid.transformer_s[t], 0)
            E_plus_l = -np.minimum(grid.lines_to_node(load_potential, (line_sallowed - grid.line_s[t])), 0)
            # E_minus_r, E_minus_l: Problems at the transformer, line from generators
            E_minus_r = -np.minimum(grid.transformer_s[t] + transformer_sallowed, 0)
            E_minus_l = -np.minimum(grid.lines_to_node(load_potential, (line_sallowed + grid.line_s[t])), 0)

            if discharging:
                return np.maximum(E_plus_r, E_plus_l) * (E_minus_l == 0)
            elif charging:
                return np.maximum(E_minus_r, E_minus_l) * (E_plus_l == 0)

    def average_charging(self, rides, t):
        """ Average charge load needed for remaining time """

        return ((self.data['energy_end'][rides['pevs']] -  # [kWh]
                 self.data['energy'][rides['pevs']]) /  # [kWh]
                (self.data['dep_t'][rides['pevs']] - t) /
                (24 / cfg.t_periods)).round(cfg.rounding)  # [kWh] -> [kW]

    def update_energy_soc_score(self, ride_data, rides, t, pev_selected, inputs, thetas, t_day):
        # Update the soc for the next period with the current charging load
        self.data['energy'][pev_selected] += (  # [kWh]
            (self.data['charge_load'][pev_selected] * (24 / cfg.t_periods)))  # [kW] -> [kWh]
        self.data['soc'][pev_selected] = (  # [%]
                self.data['energy'][pev_selected] / self.data['batt_capacity'][pev_selected])  # [kWh] -> [%]
        self.data['charge_load'][pev_selected] = 0  # [kW], reset for next period or next charging type

        if ride_data is not None:
            # Update charging score
            ride_data['cc_score'][self.smart_charger[rides['pevs']]] = \
                (thetas['tv_0'][t_day] * (ride_data['t_remain'] / cfg.t_periods) +  # More time -> lower rank (<0)
                 thetas['tv_1'][t_day] * (self.average_charging(rides, t) /
                                          inputs.p_charge)  # [kW], more -> higher rank (>0)
                 )[self.smart_charger[rides['pevs']]]
            # Update cc_max and dc_max
            # Available dis/charging capa [kW] depending on the defined SOC limits
            ride_data['cc_max'] = np.clip(((ride_data['soc_max'] - self.data['soc'][rides['pevs']]) *
                                           self.data['batt_capacity'][rides['pevs']] /
                                           (24 / cfg.t_periods)),
                                          a_min=0, a_max=None)  # [kWh] -> [kW]
            ride_data['cc_max'] = np.minimum(ride_data['cc_max'], inputs.p_charge)  # [kW]
            if inputs.charging_tech == 'V2G':
                ride_data['dc_max'] = np.clip(((ride_data['soc_min'] - self.data['soc'][rides['pevs']]) *
                                               self.data['batt_capacity'][rides['pevs']] /
                                               (24 / cfg.t_periods)),
                                              a_min=None, a_max=0)  # [kWh] -> [kW]
                ride_data['dc_max'] = np.maximum(ride_data['dc_max'], inputs.p_discharge)  # [kW]

            return ride_data, rides

    def one_timestep(self, t, pev_selected, inputs, thetas, grid, cases, curtailment_only=False, new=True,
                     new_avrg_cc=False):

        t_day = t % cfg.t_periods  # Time step within a day
        if not curtailment_only:
            # Set up the data structure for all rides in variable "rides" and "ride_data"
            rides = np.zeros(pev_selected.size,
                             dtype=list(zip(['pevs', 'nodes'],
                                            ['i', 'i'])))  # Initialize ride selection
            rides['pevs'] = pev_selected
            rides['nodes'] = self.data['node'][pev_selected]
            ride_data_dtype = \
                zip(['cc_max', 'dc_max', 'cc_score', 't_remain', 'soc_min', 'soc_max', 'cc_must'],
                    ['f2', 'f2', 'f2', 'i2', 'f2', 'f2', 'f2'])
            # Initialize array for PEV data at node
            ride_data = np.zeros(pev_selected.size, dtype=list(ride_data_dtype))

            # Get the remaining charging time for each ride and the needed soc at time t depending on it
            ride_data['t_remain'] = self.data['dep_t'][rides['pevs']] - t - 1
            ride_data['soc_min'] = np.maximum(
                ((self.data['energy_end'][rides['pevs']] -
                  ride_data['t_remain'] * inputs.p_charge * (24 / cfg.t_periods)) /
                 self.data['batt_capacity'][rides['pevs']]).clip(0),
                cfg.soc_min)
            ride_data['soc_max'] = np.maximum(
                (self.data['energy_end'][rides['pevs']] /  # [kWh]
                 self.data['batt_capacity'][rides['pevs']]).clip(0),  # [kWh] -> [%]
                cfg.soc_min)  # Only charge a maximum until energy end is reached
            if inputs.charging_tech == 'V2G':
                # For PEV that have a V2G charger the SOC can increase over actual need
                # because it can be discharged again
                ride_data['soc_max'][self.smart_charger[rides['pevs']]] = np.minimum(
                    ((self.data['energy_end'][rides['pevs']] -  # [kWh]
                      (ride_data['t_remain'] * inputs.p_discharge * (24 / cfg.t_periods))) /  # [kW] -> [kWh]
                     self.data['batt_capacity'][rides['pevs']]).clip(0),  # [kWh] -> [%]
                    cfg.soc_max)[self.smart_charger[rides['pevs']]]
            ride_data, rides = self.update_energy_soc_score(ride_data, rides, t, pev_selected, inputs, thetas, t_day)

            # Define UCC charging need based on target end energy, charge directly after arrival of PEV if UCC or as
            # last as possible if UCC_delayed
            if inputs.charging_tech == 'UCC_delayed':  # Charge as late as possible
                ride_data['cc_must'] = np.minimum(
                    np.clip(((self.data['energy_end'][rides['pevs']] - self.data['energy'][rides['pevs']]) /  # [kWh]
                             (24 / cfg.t_periods) -  # [kWh] -> [kW]
                             (ride_data['t_remain']) * inputs.p_charge),
                            a_min=0, a_max=None),
                    inputs.p_charge)  # [kW]
            else:
                ride_data['cc_must'] = np.minimum(
                    ((self.data['energy_end'][rides['pevs']] - self.data['energy'][rides['pevs']]) /  # [kWh]
                     (24 / cfg.t_periods)),  # [kWh] -> [kW]
                    inputs.p_charge)  # [kW]

            # cc_must is overwritten based on G2V and V2G strategy for PEV with smart_charger
            if inputs.charging_tech == 'G2V' or inputs.charging_tech == 'V2G':
                # Charge PEV that must dis/charge because their SOCmin/max is smaller/larger than their SOC
                # Calculate cc_must as target SOC change
                ride_data['cc_must'][self.smart_charger[rides['pevs']]] = (  # as SOC [%]
                    np.clip(ride_data['soc_min'] - self.data['soc'][rides['pevs']], a_min=0, a_max=None) +
                    np.clip(ride_data['soc_max'] - self.data['soc'][rides['pevs']], a_min=None, a_max=0)  # [%]
                    )[self.smart_charger[rides['pevs']]]
                # Convert target SOC change into charging kW
                ride_data['cc_must'][self.smart_charger[rides['pevs']]] = np.round(
                    np.clip(ride_data['cc_must'] * self.data['batt_capacity'][rides['pevs']] /  # [%] -> [kWh]
                            (24 / cfg.t_periods),  # [kWh] -> [kW]
                            a_min=inputs.p_discharge,  # [kW], limit charging to inputs.p_discharge
                            a_max=inputs.p_charge),  # [kW], limit to inputs.p_charge
                    cfg.rounding)[self.smart_charger[rides['pevs']]]
                if inputs.charging_tech == 'G2V':  # Only allow charging
                    ride_data['cc_must'] = np.clip(ride_data['cc_must'], a_min=0, a_max=None)

            cc_need_select = ride_data['cc_must'] != 0  # Select PEV that must charge
            if cc_need_select.sum() > 0:
                self.add_charging(t, np.stack([rides['pevs'],
                                               rides['nodes'],
                                               ride_data['cc_must']])[:, cc_need_select], must=True)
                self.ppev_load_must[t] = copy.deepcopy(self.ppev_load[t])  # Keep track of forced charging
                ride_data, rides = \
                    self.update_energy_soc_score(ride_data, rides, t, pev_selected, inputs, thetas, t_day)

            # Controlled charging strategy is not UCC and not UCC_delay
            if inputs.charging_tech != 'UCC' or inputs.charging_tech != 'UCC_delayed':
                # Initialize dis-/charging capacity as zero
                self.fleet_cc_capa[t, :] = np.zeros(len(grid.nodes['n_id']))  # [kW]
                self.fleet_dc_capa[t, :] = np.zeros(len(grid.nodes['n_id']))  # [kW]
                # Aggregate dis/-charging capacities per node, only data from PEV that do not need to charge is
                # utilized
                capa_per_node = quick_groupby(rides['nodes'],
                                              np.vstack([ride_data['cc_max'], ride_data['dc_max']]).transpose(),
                                              np.sum)  # index, data [kW]
                self.fleet_cc_capa[t, rides['nodes']] = capa_per_node[:, 0]  # [kW]
                self.fleet_dc_capa[t, rides['nodes']] = capa_per_node[:, 1]  # [kW]

                # Update load_case and transformer_s based on c_must = ppev_load
                grid.transformer_s[t], grid.line_s[t] = grid.update_load_case(pev_load=self.ppev_load[t],
                                                                              t=t, get_currents_s=True)  # [kW]
                # Charge in the feedin-case if load at node are too high
                overload = self.get_node_allowed(grid, self.fleet_cc_capa[t, :], t, cases, charging=True)  # Overload
                if inputs.approach == 'budget':
                    self.original_load_delta[t] = np.minimum(overload,
                                                             self.fleet_cc_capa[t] * thetas['tc_0'][t_day])  # Budget
                else:
                    self.original_load_delta[t] = overload * thetas['tc_0'][t_day]

                # Discharge in the load-case if load at node are too high
                if inputs.charging_tech == 'V2G':
                    overload = self.get_node_allowed(grid, self.fleet_dc_capa[t, :], t, cases,
                                                     discharging=True)  # Overload
                    if inputs.approach == 'budget':
                        self.original_load_delta[t] = np.minimum(overload,
                                                                 self.fleet_dc_capa[t] * thetas['td_0'][t_day])  # Budg.
                    else:
                        self.original_load_delta[t] += -overload * thetas['td_0'][t_day]

                self.remaining_load_delta[t] = copy.deepcopy(self.original_load_delta[t])  # [kW]
                # G2V case
                if self.remaining_load_delta[t, rides['nodes']].sum() > 0:  # Only charge load_delta that are positive
                    ride_data, rides = self.targeted_charging(ride_data, rides, t, self.remaining_load_delta[t],
                                                              new=new)
                # V2G case
                if (self.remaining_load_delta[t, rides['nodes']].sum() < 0 and  # Only discharge load_delta are negative
                        inputs.charging_tech == 'V2G'):  # Discharge if V2G is enabled
                    ride_data, rides = self.targeted_charging(ride_data, rides, t, self.remaining_load_delta[t],
                                                              discharge=True, new=new)
                # Keep track of smart charging
                self.ppev_load_smart[t] = self.ppev_load[t] - self.ppev_load_must[t]
                ride_data, rides = \
                    self.update_energy_soc_score(ride_data, rides, t, pev_selected, inputs, thetas, t_day)

                # Set RIDE_DATA: [kW] to average charge load and load delta to free trafo capacity
                if cfg.average_charging:  # and rides['pevs'].size != 0:
                    # Distribute free load from lines based on cc_average
                    load_potential = np.zeros(len(grid.nodes['n_id']))
                    load_potential[rides['nodes']] = quick_groupby(rides['nodes'],
                                                                   np.clip(self.average_charging(rides, t),
                                                                           a_max=None, a_min=0),
                                                                   np.sum)

                    # Update load_case and transformer_s based on c_must = ppev_load
                    grid.transformer_s[t], grid.line_s[t] = grid.update_load_case(pev_load=self.ppev_load[t],
                                                                                  t=t, get_currents_s=True)
                    free = self.get_node_allowed(grid, load_potential, t, cases, charging=True, free=True)  # Free
                    if inputs.approach == 'budget':
                        node_free = np.minimum(free,
                                               load_potential * thetas['tc_1'][t_day])  # Budget
                    else:
                        node_free = free * thetas['tc_1'][t_day]

                    # Fully charge all PEV until node_free is reached
                    avrg_factor = thetas['tc_1'][t_day] if new_avrg_cc else 1.0
                    # Replace ride_data['cc_max'] with average charging * factor
                    ride_data['cc_max'] = self.average_charging(rides, t) * avrg_factor
                    ride_data, rides = self.targeted_charging(ride_data, rides, t, node_free,
                                                              new=new, avrg_factor=avrg_factor)

                    average_discharging = False
                    if inputs.charging_tech == 'V2G' and average_discharging:
                        # Distribute free load from lines based on cc_average
                        load_potential = np.zeros(len(grid.nodes['n_id']))
                        load_potential[rides['nodes']] = quick_groupby(rides['nodes'],
                                                                       np.clip(self.average_charging(rides, t),
                                                                               a_max=0, a_min=None),
                                                                       np.sum)

                        # Update load_case and transformer_s based on c_must = ppev_load
                        grid.transformer_s[t], grid.line_s[t] = grid.update_load_case(pev_load=self.ppev_load[t],
                                                                                      t=t, get_currents_s=True)
                        if inputs.approach == 'budget':
                            free = self.get_node_allowed(grid, load_potential, t, cases, discharging=True, free=True)
                            node_free = np.minimum(free,
                                                   load_potential * thetas['td_1'][t_day])  # TODO: Min oder Max?
                        else:
                            node_free = self.get_node_allowed(grid, load_potential, t, cases,
                                                              discharging=True, free=True
                                                              ) * thetas['td_1'][t_day]

                        # Fully discharge all PEV until node_free is reached
                        ride_data, rides = self.targeted_charging(ride_data, rides, t, node_free,
                                                                  discharge=True, new=new)

                    # Keep track of average charging
                    self.ppev_load_avrg[t] = self.ppev_load[t] - self.ppev_load_must[t] - self.ppev_load_smart[t]

                # Update capacity at transformers and lines after final charging
                grid.transformer_s[t], grid.line_s[t] = grid.update_load_case(pev_load=self.ppev_load[t],
                                                                              t=t, get_currents_s=True)

            self.update_energy_soc_score(None, rides, t, pev_selected, inputs, thetas, t_day)

        if inputs.curtailment:
            if curtailment_only:
                # Update capacity at transformers and lines
                grid.transformer_s[t], grid.line_s[t] = grid.update_load_case(pev_load=self.ppev_load[t],
                                                                              t=t, get_currents_s=True)
            overload = self.get_node_allowed(grid, grid.pp_load[t, :] * 1e3, t, cases, charging=True)  # kW
            if inputs.approach == 'budget':
                curtailment_delta_t = np.minimum(overload,
                                                 grid.pp_load[t, :] * 1e3 * thetas['tr_0'][t_day])  # MW->kW, Budget
            else:
                curtailment_delta_t = overload * thetas['tr_0'][t_day]
            grid.pcurtail[t] = np.minimum(grid.pp_load[t],
                                          curtailment_delta_t / 1000)
            self.remaining_load_delta[t] -= grid.pcurtail[t] * 1000  # [MW] -> [kW]

    def check_chargers(self, tt, n, missing, pev_select, smart=False):

        def remove(pev_sel, del_pev, pev_at_n, miss):
            pev_no_cc = np.random.choice(pev_at_n,
                                         min(int(miss), len(pev_at_n)), replace=False)
            if smart:
                self.smart_charger[pev_no_cc] = False
            else:
                pev_sel = np.delete(pev_sel, [np.where(pev_sel == x)[0][0] for x in pev_no_cc])
            del_pev += len(pev_no_cc)
            miss -= len(pev_no_cc)

            return pev_sel, del_pev, miss

        deleted_pev, t_past = 0, 0
        while deleted_pev < int(missing[n]):
            # Select PEV that do not get a charger, initially only PEV without current charging need
            pev_at_node = pev_select[
                (self.data['node'][pev_select] == n) *  # PEV at node
                (self.data['energy_tb_charged'][pev_select] == 0)]  # PEV w/o charging
            pev_at_node_t = pev_at_node[self.data['arr_t'][pev_at_node] == tt - t_past]
            if smart:
                pev_at_node_t = pev_at_node_t[np.where(self.smart_charger[pev_at_node_t])]
            if len(pev_at_node_t) < int(missing[n]):  # Add PEV with charging need
                pev_select, deleted_pev, missing[n] = remove(pev_select, deleted_pev, pev_at_node_t, missing[n])
                pev_at_node = pev_select[self.data['node'][pev_select] == n]
                pev_at_node_t = pev_at_node[self.data['arr_t'][pev_at_node] == tt - t_past]
                if smart:
                    pev_at_node_t = pev_at_node_t[np.where(self.smart_charger[pev_at_node_t])]

            pev_select, deleted_pev, missing[n] = remove(pev_select, deleted_pev, pev_at_node_t, missing[n])
            t_past += 1

        if not smart:
            return pev_select

    def calc_charging_curtailment(self, grid, inputs, thetas, new=True, new_avrg_cc=False):
        """Calculated the vehicle specific states for all timesteps given a DistributionGrid object, also draws the
        resulting charging action and calculated SOC and energy development for all timesteps. Variables:
            - Scv being the charging score for each vehicle,
            - Tcv the deterministic charging threshold for vehicle v, and
            - M is a large number,
            - Tv is the time that vehicle v will be plugged,
            - Eplustv is the maximal energy that can be charged in the vehicle,
            - Eminustv is the maximal energy that can be discharged from the vehicle and
            - Dvis the depth of discharge occurring if the vehicle will discharge in the current period t.
        Heaviside function: https://docs.scipy.org/doc/numpy/reference/generated/numpy.heaviside.html"""
        import time

        def empty_array(rows=inputs.index.size, columns=len(grid.nodes['node']), data_type='float32'):
            return np.zeros((rows, columns)).astype(data_type)

        logger.info('Starting the calculation of vehicle specific states')
        self.ppev_load = empty_array()  # [kW]
        self.ppev_load_must = empty_array()  # [kW]
        self.ppev_load_smart = empty_array()  # [kW]
        self.ppev_load_avrg = empty_array()  # [kW]
        self.fleet_cc_capa = empty_array()  # [kWh]
        self.fleet_dc_capa = empty_array()  # [kWh]
        self.original_load_delta = empty_array()  # [kW]
        self.remaining_load_delta = empty_array()  # [kW]
        if inputs.charging_tech == 'V2G':
            self.charging_history = np.zeros((self.data.shape[0], cfg.t_periods)).astype('float16')  # [kW]
        else:
            self.charging_history = None
        grid.transformer_s = empty_array()
        grid.line_s = empty_array(columns=len(grid.line_selections[0]))
        grid.pcurtail = empty_array()
        cases = grid.edisgo.network.timeseries.timesteps_load_feedin_case

        # Seed to avoid choosing episode where smart_cc are distributed most favourably
        # np.random.seed(inputs.edisgo_grid)
        self.smart_charger = np.random.choice([False, True], self.data.shape[0],
                                              p=[1 - self.charger_attr['smart_cc_share'],
                                                 self.charger_attr['smart_cc_share']])

        start = time.time()
        for t in range(inputs.index.size):   # [2] = arrival time, [5] = departure time
            if t % 5000 == 0:
                logger.info('Charging and curtailment in grid {} for timestep {}/{} - runtime = {}'.format(
                    inputs.edisgo_grid, t, inputs.index.size, round(time.time() - start, 2)))

            # Check if charger is available for each PEV, if less chargers than PEV arriving then less PEV are
            # considered in the current period and postponed to other periods
            pev_select = self.get_pev_select(inputs, grid, t)
            if pev_select.shape[0] > 0:  # Continue if PEV are available
                nodes_select, pev_cc_nodes = np.unique(
                    self.data['node'][pev_select], return_counts=True)
                missing_cc = np.zeros(len(grid.nodes['node']))

                missing_cc[nodes_select] = np.clip(pev_cc_nodes -
                                                   self.number_chargers['chargers'][nodes_select],
                                                   a_min=0, a_max=None)

                if missing_cc.sum() > 0:
                    nodes_missing_chargers = np.where(missing_cc > 0)[0]
                    logger.debug('Time {}: Number of missing chargers = {} at nodes {}'.format(
                         t, missing_cc.sum(), nodes_missing_chargers))
                    for node in nodes_missing_chargers:
                        pev_select = self.check_chargers(t, node, missing_cc, pev_select)

                # Check if smart charger is available for each PEV
                # How can pev.smart_charger.sum() != pev.number_chargers['smart_chargers'].sum()
                # -> the first is for nodes and the second for rides
                nodes_select, pev_smartcc_nodes = np.unique(
                    self.data['node'][pev_select[np.where(self.smart_charger[pev_select])]], return_counts=True)
                missing_smartcc = np.zeros(len(grid.nodes['node']))
                missing_smartcc[nodes_select] = np.clip(pev_smartcc_nodes -
                                                        self.number_chargers['smart_chargers'][nodes_select],
                                                        a_min=0, a_max=None)

                if missing_smartcc.sum() > 0:
                    nodes_missing_chargers = np.where(missing_smartcc > 0)[0]
                    logger.debug('Time {}: Number of missing smart chargers = {} at nodes {}'.format(
                        t, missing_smartcc.sum(), nodes_missing_chargers))
                    for node in nodes_missing_chargers:
                        self.check_chargers(t, node, missing_smartcc, pev_select, smart=True)

                if inputs.charging_tech == 'G2V':
                    # Exclude PEV that have no charging need because nothing will be done with those anyway in G2V
                    pev_select = pev_select[self.data[pev_select]['energy_tb_charged'] != 0]

                if pev_select.size > 0:
                    self.one_timestep(t, pev_select, inputs, copy.deepcopy(thetas), grid, list(cases.iloc[t]),
                                      new_avrg_cc=new_avrg_cc)
                else:
                    if inputs.curtailment:
                        self.one_timestep(t, pev_select, inputs, copy.deepcopy(thetas), grid, list(cases.iloc[t]),
                                          curtailment_only=True, new_avrg_cc=new_avrg_cc)
            else:
                if inputs.curtailment:
                    self.one_timestep(t, pev_select, inputs, copy.deepcopy(thetas), grid, list(cases.iloc[t]),
                                      curtailment_only=True, new_avrg_cc=new_avrg_cc)

        # Reduce size of self.charging_history
        if inputs.charging_tech == 'V2G':
            x, y = self.charging_history.nonzero()
            self.charging_history = [x, y, self.charging_history[x, y]]


def update_number_chargers(inputs_dict, number_of_sets, testing=False):
    import dill

    if testing:
        import re

        # For out of sample testing, get number of chargers from in-sample scenarios
        inputs = inputs_dict[list(inputs_dict.keys())[0]]
        optimization_episode = re.split('_e', inputs.extreme_file['base_ucc'])[1][:4]
        worstuccpevfile = '{}/PEV_[{}]_pevsobject_p{}_y{}_{}_soc{}{}.pkl'.format(
            re.split('PEV_', inputs.pev_file)[0],
            inputs.edisgo_grid,
            inputs.pev_share[inputs.year],
            inputs.year,
            optimization_episode,
            inputs.socinit_min,
            inputs.socinit_max)
        with open(worstuccpevfile, 'rb') as f:
            temp = dill.load(f)
        chargers = temp.number_chargers['chargers']
    else:
        # For in-sample scenarios, get maximum chargers across all scenarios
        with open(inputs_dict[list(inputs_dict.keys())[0]].pev_file, 'rb') as f:
            temp = dill.load(f)
        number_nodes = len(temp.number_chargers['chargers'])

        # Get maximum number of chargers across all scenarios
        ccs = np.zeros((number_of_sets, number_nodes))
        i = 0
        for key in inputs_dict.keys():
            with open(inputs_dict[key].pev_file, 'rb') as f:
                temp = dill.load(f)
            ccs[i] = temp.number_chargers['chargers']
            i += 1
        chargers = np.max(ccs, axis=0)  # ccs.max(0)

    # Set maximum number of chargers for all scenarios
    for key in inputs_dict.keys():
        with open(inputs_dict[key].pev_file, 'rb') as f:
            temp = dill.load(f)
        if np.all(temp.number_chargers['chargers'] == chargers):  # (temp.number_chargers['chargers'] == chargers).all()
            pass
        else:
            temp.number_chargers['chargers'] = chargers
            temp.number_chargers['smart_chargers'] = chargers
            with open(inputs_dict[key].pev_file, 'wb') as f:
                dill.dump(temp, f)


def get_pevdata():
    import dill

    # Get PEV data
    ride_prob, ta, td, km_from, km_to = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for load_type in cfg.load_types:
        input_file = '{}/Inputs/PEV_data_{}.pkl'.format(cfg.parent_folder, load_type)
        with open(input_file, 'rb') as fff:
            ride_prob_data, ta_data, td_data, km_from_data, km_to_data = dill.load(fff)

        # Add load_type as columns
        ride_prob_data.loc[:, 'load_type'] = load_type
        ta_data.loc[:, 'load_type'] = load_type
        td_data.loc[:, 'load_type'] = load_type
        km_from_data.loc[:, 'load_type'] = load_type
        km_to_data.loc[:, 'load_type'] = load_type

        # Append to existing load_types data
        ride_prob = ride_prob.append(ride_prob_data)
        ta = ta.append(ta_data)
        td = td.append(td_data)
        km_from = km_from.append(km_from_data)
        km_to = km_to.append(km_to_data)

    return ride_prob, ta, td, km_from, km_to


def unique1d(ar):
    """Find the index of the unique elements of an array, ignoring shape. """
    ar = np.asanyarray(ar).flatten()
    perm = ar.argsort(kind='quicksort')
    aux = ar[perm]
    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]
    inv_idx = np.empty(mask.shape, dtype=np.intp)
    inv_idx[perm] = np.cumsum(mask) - 1

    return inv_idx


def quick_groupby(index, data, function_):
    """Takes index and data array as inputs and returns a new array grouped by the index and cumsumed
     Refer to http://esantorella.com/2016/06/16/groupby/"""

    keys_as_int = unique1d(np.asanyarray(index))  # 50% der Zeit
    x = np.array([range(len(keys_as_int)), keys_as_int])  # Teilweise 70% der zeit
    result = np.zeros(data.shape)  # , order='c')
    for i in np.unique(keys_as_int):  # 40% der Zeit
        idx = np.array(x[0][x[1] == i])
        result[idx] = function_(data[idx], axis=0)

    return result


# Test if this really works before using it!
def quick_groupby2(index, data, function_):
    """Takes index and data array as inputs and returns a new array grouped by the index and cumsumed
     Refer to http://esantorella.com/2016/06/16/groupby/"""
    index = index.astype('int')
    result = np.zeros(data.shape)  # , order='c')
    for i in np.unique(index):
        idx = np.where(index == i)[0]
        result[idx] = function_(data[idx], axis=0)

    return result


def pev_model(node, day, pev_number, load_type, inputs, ride_prob, ta, td, km_to, dtype, gmp_data, new_rides=False):
    """Simulate PEV: Randomly selects km, arrival and departure times for PEV at all nodes n
    PEV model output:
    # [0] = ride ID, [1] = ride in a day ID, [2] = arr_t, [3] = batt_capacity, [4] = demand_type, [5] = dep_t,
    # [6] = energy_start, [7] = energy_tb_charged [kWh], [8] = node, [9] = energy_end, [10] = soc,
    # [11] = energy [kWh], [12] = charge_load [kW], [13] = day number, [14] = charged [kWh], [15] = discharged [kWh]
    """

    # Seed based on episode, node, load type and day:
    # load_types = {'agricultural': 1, 'industrial': 2, 'residential': 3, 'retail': 4}
    # seed = int(inputs.episode * (node + 1) * load_types[load_type] * (np.where(inputs.index == day)[0][0] + 1) / 1e5)
    seed = None
    # np.random.seed(seed)

    pev = np.zeros(pev_number, dtype=dtype)
    pev['node'] = node
    pev_makes = dict(
        Make=["small PHEV", "mid PHEV", "large PHEV", "small BEV", "mid BEV", "large BEV"],
        Share=list((np.array(cfg.s_m_l_shares) * np.array(cfg.bev_phev_shares[inputs.year]))),
        BTC=cfg.battery_capa[inputs.scenario_battery][inputs.year],
        EU=cfg.energy_usage[inputs.scenario_energyuse][inputs.year])

    # Select PEV Model
    pev_select = choices(list(range(len(pev_makes['Share']))),
                         pev_makes['Share'],
                         k=pev_number)

    # Select reference SOC from normal dist, Keep only 0% <= SOC <= 100%, Redraw otherwise

    while pev['soc'].max() >= inputs.socinit_max or pev['soc'].min() <= inputs.socinit_min:
        pev['soc'][pev['soc'] >= inputs.socinit_max] = \
            np.random.normal(cfg.soc_mean, cfg.soc_std,
                             size=np.sum(pev['soc'] >= inputs.socinit_max))  # .sum())
        pev['soc'][pev['soc'] <= inputs.socinit_min] = \
            np.random.normal(cfg.soc_mean, cfg.soc_std,
                             size=np.sum(pev['soc'] <= inputs.socinit_min))  # .sum())

    logger.debug('Node {}, Day {} - min: {}, max: {} with seed {}'.format(
        node, day, pev['soc'].min(), pev['soc'].max(), seed))

    pev['batt_capacity'] = np.array(pev_makes['BTC'])[pev_select]
    pev['day'] = inputs.day_number

    if new_rides:
        gmp_zweck = gmp_data[gmp_data.ZWECK.isin(cfg.zweck_per_area[load_type])]
        gmp_select = gmp_zweck.sample(n=pev_number, replace=True)
        pev['energy_tb_charged'] = gmp_select.KM_NEXT * np.array(pev_makes['EU'])[pev_select] / 100  # km/eu*100
        pev['arr_t'] = gmp_select.ANZEIT
        pev['dep_t'] = gmp_select.ABZEIT_STOP + (gmp_select.WOTAG_STOP - gmp_select.WOTAG) * cfg.t_periods
        # IF PEV are staying from one week to another
        pev['dep_t'][pev['dep_t'] < 0] = \
            (gmp_select.ABZEIT_STOP + (gmp_select.WOTAG_STOP + 7 - gmp_select.WOTAG) * cfg.t_periods)[pev['dep_t'] < 0]
    else:
        pev['energy_tb_charged'] = choices(km_to.Value.tolist(), km_to.Probability.tolist(), k=pev_number
                                           ) * np.array(pev_makes['EU'])[pev_select] / 100  # km/eu*100

        # Set plugin time (trip arrival time), end of plugin time (trip departure time) and trip distance (km)
        pev['arr_t'] = choices(ta.Value.tolist(), ta.Probability.tolist(), k=pev_number)
        pev['dep_t'] = choices(td.Value.tolist(), td.Probability.tolist(), k=pev_number)
        pev['dep_t'] += (pev['dep_t'] < pev['arr_t']) * cfg.t_periods  # If td < ta then set to next day
        # Should PEV stay at least an hour?

    # Update to current day
    pev['arr_t'] += pev['day'] * cfg.t_periods
    pev['dep_t'] += pev['day'] * cfg.t_periods

    if load_type == 'residential':  # Trips of pev that do not have a ride on a given day at residential nodes
        # Calculate ride or no ride
        ride = np.array(choices([0, 1],
                                [(1 - ride_prob.Avg_Ride_probability_per_car), ride_prob.Avg_Ride_probability_per_car],
                                k=pev_number))
        if inputs.date != '2011-01-01':
            pev['arr_t'] -= (1 - ride) * cfg.t_periods  # Arriving a day before
        if inputs.date != '2011-12-31':
            pev['dep_t'] += (1 - ride) * cfg.t_periods  # Leaving the next day
        pev['energy_tb_charged'] = ride * pev['energy_tb_charged']

    # Remove energy needs that are larger than the battery capacity
    pev['energy_tb_charged'] = np.minimum(pev['energy_tb_charged'],
                                          pev['batt_capacity'] - pev['soc'] * pev['batt_capacity'])

    return pev


def calc_pev_data(inputs, pev_number, grid, ride_prob, ta, td, km_to, number_rides_data, ride_purpose_data, gmp_data,
                  dtype, day, return_rides_at_node=False, new_rides=False):
    inputs.update_day(str(day.date()))
    number_rides_data = number_rides_data[number_rides_data.WOTAG == inputs.weekday]
    ride_purpose_data = ride_purpose_data[ride_purpose_data.WOTAG == inputs.weekday]
    ride_prob = ride_prob.loc[inputs.weekday].set_index('load_type')
    ta = ta[ta.WOTAG == inputs.weekday].set_index('load_type')
    td = td[td.WOTAG == inputs.weekday].set_index('load_type')
    km_to = km_to[km_to.WOTAG == inputs.weekday].set_index('load_type')
    gmp_data = gmp_data[gmp_data.WOTAG == inputs.weekday]

    number_rides = choices(number_rides_data.Value.tolist(),
                           number_rides_data.Probability.tolist(),
                           k=pev_number)  # Number of rides per PEV from residential node
    count_ride_purpose = pd.Series()
    ride_purpose = choices(ride_purpose_data.Value.tolist(),
                           ride_purpose_data.Probability.tolist(),
                           k=sum(number_rides))  # Purpose per ride from residential node
    count_ride_purpose.loc['retail'] = ride_purpose.count(4) + ride_purpose.count(11)
    count_ride_purpose.loc['ind_ret_agr'] = ride_purpose.count(1) + ride_purpose.count(3)
    count_ride_purpose.loc['other'] = ride_purpose.count(5) + ride_purpose.count(9)
    assert count_ride_purpose.sum() == sum(number_rides)  # Check if all rides have been recorded

    # Distribute rides across all nodes
    # Number of employees per int/ret/agr load at nodes
    employees = grid.consumption.mul(cfg.employees, axis=0
                                     ).round(0).fillna(0).astype(int)
    rides_at_nodes = employees / employees.sum().sum() * count_ride_purpose.ind_ret_agr
    rides_at_nodes.loc['retail', :] += (employees.loc['retail', :] /
                                        employees.loc['retail', :].sum() *
                                        count_ride_purpose.retail)
    rides_at_nodes.loc['residential', :] += (grid.consumption.loc['residential', :] /
                                             grid.consumption.loc['residential', :].sum() *
                                             pev_number)
    rides_at_nodes = rides_at_nodes.round(0).fillna(0).astype(int)
    if return_rides_at_node:
        return rides_at_nodes

    rawdata = np.zeros(0, dtype=dtype)  # Initialize DataFrame for PEV data at node
    for n_id in grid.nodes['n_id']:
        node = grid.nodes['node'][n_id]
        for load_type in cfg.load_types:  # Generate PEV driving pattern for each load type at each node
            try:
                rides = rides_at_nodes.loc[load_type, node]
                if rides > 0:
                    pev = pev_model(n_id, day, rides, load_type, inputs, ride_prob.loc[load_type],
                                    ta.loc[load_type], td.loc[load_type], km_to.loc[load_type], dtype, gmp_data,
                                    new_rides)
                    rawdata = np.append(rawdata, pev, axis=0)
                    logger.debug('Generated {} PEV rides for node {}, {} and day {}'.format(
                        rides_at_nodes.loc[load_type, node], node, load_type, inputs.date))
            except KeyError:
                logger.debug('No rides at node {} and load type {}'.format(node, load_type))

    return rawdata


def get_pev(inputs, smart_cc_share=1.0, return_bool=True, grid=None, new_rides=False):
    """Initialize PEV and get results from multiprocessing into dictionaries with entries for each day"""
    import os
    import dill
    import multiprocessing as mp
    from Functions import grid_functions as grd
    from functools import partial

    if not os.path.isfile(inputs.pev_file):
        logger.warning('Generating PEV scenario for {} on process {} for ... {}'.format(
            inputs.episode, os.getpid(), inputs.pev_file[-50:]))

        # Seed for consistent selection between different tests
        # np.random.seed(inputs.episode)

        # Get PEV data
        if grid is None:
            grid = grd.get_grid(inputs)  # Get grid object
        ride_prob, ta, td, km_from, km_to = get_pevdata()
        gmp_data = pd.read_csv('{}/Inputs/GMP_data_modified.csv'.format(cfg.parent_folder))

        # Get number of rides from residential node
        with open('{}/Inputs/number_rides_from_RES.pkl'.format(cfg.parent_folder), 'rb') as f:
            number_rides_data = dill.load(f)

        # Get ride purpose data from residential node
        with open('{}/Inputs/zweck_rides_from_RES.pkl'.format(cfg.parent_folder), 'rb') as f:
            ride_purpose_data = dill.load(f)

        pev_numbers = int(round(grid.pkw_number * inputs.pev_share[inputs.year], 0))
        dtype = list(
            zip(['arr_t', 'batt_capacity', 'dep_t', 'energy_tb_charged', 'node', 'energy_end', 'soc',
                 'energy', 'charge_load', 'day', 'charged', 'discharged'],
                ['i4', 'i2', 'i4', 'f2', 'i2', 'f2', 'f2', 'f2', 'f2', 'i2', 'f2', 'f2']))
        rawdata = np.zeros(0, dtype=dtype)  # Initialize DataFrame for PEV data at node

        if inputs.parallel_processing:
            pools = int(min(mp.cpu_count(), len(inputs.dayindex)))
            pool = mp.Pool(processes=pools)  # Spawn a process for each day
            func = partial(calc_pev_data, inputs, pev_numbers, grid, ride_prob, ta, td, km_to, number_rides_data,
                           ride_purpose_data, gmp_data, dtype, new_rides=new_rides)
            output = pool.map(func, inputs.dayindex)
            pool.close()
            pool.join()
            rawdata = pd.concat(output, ignore_index=True)
        else:
            for day_p in inputs.dayindex:
                pev = calc_pev_data(inputs, pev_numbers, grid, ride_prob, ta, td, km_to, number_rides_data,
                                    ride_purpose_data, gmp_data, dtype, day_p, new_rides=new_rides)
                rawdata = np.append(rawdata, pev, axis=0)
                logger.debug('PEV calculated for grid {} and day {}, current total number of rides: {}'.format(
                    inputs.edisgo_grid, day_p, rawdata.shape))

        rawdata['dep_t'] = np.clip(rawdata['dep_t'],
                                   a_min=0,
                                   a_max=(365 * cfg.t_periods - 1))  # Limit time to current year
        # Avoid that energy_end < soc_min * battery_capacity
        rawdata['energy'] = rawdata['soc'] * rawdata['batt_capacity']
        rawdata['energy_end'] = np.maximum(rawdata['batt_capacity'] * cfg.soc_min,
                                           rawdata['energy'] + rawdata['energy_tb_charged'])
        rawdata['energy_tb_charged'] = rawdata['energy_end'] - rawdata['energy']

        pevs_year = PEVInfo(rawdata, inputs)
        pevs_year.get_number_chargers(smart_cc_share, cfg.charger_availability, inputs, grid, first=True)
        pevs_year.node_batt_capa = np.array(
            [np.sum(pevs_year.data['batt_capacity'][pevs_year.data['node'] == n]) for n in
             range(len(grid.nodes['node']))],
            dtype='int')
        with open(inputs.pev_file, 'wb') as f:
            dill.dump(pevs_year, f)

        logger.info('PEV for grid {} generated and saved, size of rawdata for rides: {}'.format(
            inputs.edisgo_grid, rawdata.shape))

        cfg.aws_upload_results(inputs.name, grid_data=True, file=inputs.pev_file)

        if return_bool:
            return pevs_year
    else:
        logger.warning('PEV scenario for {} already available in  ... {}'.format(
            inputs.episode, inputs.pev_file[-50:]))

        if return_bool:
            logger.info('Getting PEV object from file ...{}'.format(inputs.pev_file[-50:]))

            with open(inputs.pev_file, 'rb') as f:
                pevs_year = dill.load(f)
            pevs_year.get_number_chargers(smart_cc_share, cfg.charger_availability, inputs, grid)

            return pevs_year


def get_set_of_pev(inputs_dict, max_parallel=10, new_rides=False):
    import multiprocessing as mp
    from functools import partial

    logger.info('Generating PEV scenarios for the following inputs: \n{}'.format(inputs_dict))

    # Calculate sets of PEV
    pool = mp.Pool(processes=min(max_parallel, len(inputs_dict)),
                   maxtasksperchild=1)  # mp.cpu_count())  # Spawn processes
    func = partial(get_pev, new_rides=new_rides,
                   return_bool=False)
    pool.map(func, list(inputs_dict.values()), chunksize=1)
    pool.close()
    pool.join()


def get_data_from_gmp():
    df = pd.read_csv('GMP_data.csv', sep=';')
    df.loc[:, 'ID_PERS'] = ['{}_{}'.format(df.loc[x, 'ID'], df.loc[x, 'PERSNR']) for x in df.index]
    df.loc[:, 'ABZEIT_STOP'] = [0 for x in df.index]
    df.loc[:, 'WOTAG_STOP'] = [0 for x in df.index]
    df.loc[:, 'KM_NEXT'] = [0 for x in df.index]
    df.drop('ID', inplace=True, axis=1)
    df.drop('PERSNR', inplace=True, axis=1)
    df.drop('DAUER', inplace=True, axis=1)

    data_new = pd.DataFrame()
    for id_pers in set(df.ID_PERS):
        logger.info('Working on getting data for Person {}'.format(id_pers))
        data = df.query('ID_PERS == "{}"'.format(id_pers))
        ABZEIT_STOP = [data.ABZEIT.iloc[x + 1] for x in range(len(data.index) - 1)]
        WOTAG_STOP = [data.WOTAG.iloc[x + 1] for x in range(len(data.index) - 1)]
        KM_NEXT = [data.KM.iloc[x + 1] for x in range(len(data.index) - 1)]
        data = data.iloc[:-1, :]
        data.ABZEIT_STOP = ABZEIT_STOP
        data.WOTAG_STOP = WOTAG_STOP
        data.KM_NEXT = KM_NEXT
        data.drop('ABZEIT', inplace=True, axis=1)
        data_new = data_new.append(data)

    data_new.ABZEIT_STOP = data_new.ABZEIT_STOP.astype(int)
    data_new.WOTAG_STOP = data_new.WOTAG_STOP.astype(int)
    data_new.KM_NEXT = data_new.KM_NEXT.astype(int)

    data_new.ANZEIT = [int(round(x * 48 / 2400, 0)) for x in data_new.ANZEIT]
    data_new.ABZEIT_STOP = [int(round(x * 48 / 2400, 0)) for x in data_new.ABZEIT_STOP]
