import os
import numpy as np
import pandas as pd
from Functions import grid_functions as grd, config_functions as cfg, loadflow_functions as lff

import logging
logger = logging.getLogger(__name__)


class CostCalc:
    """
    Define a DataFrame for all cost types (columns) and all years (rows) included costs are:
        'annuities_grid' = Distribution grid investments in annuities
        'cost_curtail' = The cost of lost energy through curtailment
        'cost_charge_loss' = The cost of lost energy through charging inefficiencies from discharging PEV
        'cost_battery_degr' = The cost of battery degradation when PEV are discharged
        'annuities_charge' = The investments for smart PEV chargers in annuities
        'cost_communication' = The cost of communication with smart PEV chargers (monthly fees)
        'cost_grid_losses' = The cost of lost energy within the distribution grid components
    """
    __slots__ = ('cost', 'cost_periods', 'investments_grid', 'investments_pev')

    def __init__(self, inputs, episode=None):
        self.cost = pd.Series(0, index=cfg.cost_types, name=inputs.episode if episode is None else episode)
        self.cost_periods = pd.DataFrame(0.0, index=inputs.index, columns=cfg.cost_types)
        self.investments_grid = 0
        self.investments_pev = 0

    def investment_needs(self, grid, ppev_load, inputs, without_generator_import=True, combined_expansion=False):
        from edisgo.flex_opt import exceptions

        try:
            if not combined_expansion:
                grid.get_reinforcement_timesteps(inputs, ppev_load)

            # grid.timesteps['initial'] = grid.timesteps['initial'][:20]
            # Grid reinforcement incl. load flow
            grid = lff.reinforce_grid(inputs,
                                      grid=grid,
                                      timesteps_pfa=grid.timesteps['initial'],
                                      max_while_iterations=cfg.lf_iterations,
                                      without_generator_import=without_generator_import,
                                      mode='mv',
                                      linear=cfg.linear_pf,
                                      copy_graph=False)  # Used to be True and grid.edisgo passed

            self.investments_grid = \
                int(grid.edisgo.network.results.grid_expansion_costs.total_costs.sum() * 1e3)  # kEUR to EUR

            if self.investments_grid > 0:
                invest = pd.DataFrame(index=grid.edisgo.network.results.grid_expansion_costs.index,
                                      columns=['type', 'cost'])
                try:
                    invest.type = ['trafo' if x else 'line' for x in
                                   grid.edisgo.network.results.grid_expansion_costs.length.isnull()]
                except AttributeError:  # If no lines need to be expanded "length"-values is not available
                    invest.type = ['trafo' for _ in
                                   grid.edisgo.network.results.grid_expansion_costs.index]
                invest.cost = grid.edisgo.network.results.grid_expansion_costs.total_costs * 1e3
                invest = invest.groupby('type').sum().cost
                try:
                    ann_lines = invest.line * cfg.annuity_factor(cost_type='line')
                except AttributeError:
                    ann_lines = 0
                try:
                    ann_trafo = invest.trafo * cfg.annuity_factor(cost_type='trafo')
                except AttributeError:
                    ann_trafo = 0
                self.cost.annuities_grid = round(ann_lines + ann_trafo, 2)
            else:
                self.cost.annuities_grid = 0

            logger.info('\n=== Analyzed (reinforced) grid with PEV load (and curtailment), grid expansion'
                        'investment needs: {} ==='.format(int(self.investments_grid)))
            self.cost_periods.annuities_grid = pd.Series(self.cost.annuities_grid / cfg.t_periods,
                                                         index=inputs.index)
        except (ValueError, SystemError, RuntimeError, AttributeError, exceptions.MaximumIterationError) as e:
            logger.error('Error in the power flow analysis: {}: {}'.format(type(e), e))
            self.investments_grid = 99999999  # kEUR to EUR
            self.cost.annuities_grid = 9999999  # annuity factor
            self.cost_periods.annuities_grid = pd.Series(self.cost.annuities_grid / cfg.t_periods,
                                                         index=inputs.index)

        return grid

    def curtailment_cost(self, grid, inputs, curtailed_per_unit):
        import re

        curt_cost_generators = pd.Series(index=grid.edisgo.network.pypsa.generators.type.index)
        for n_full in grid.edisgo.network.pypsa.generators.type.index:
            try:
                gen_type = grid.edisgo.network.pypsa.generators.type.loc[n_full]
                gen_type = re.split('_', gen_type)[0] if gen_type is not '' else re.split('_', n_full)[0]
                curt_cost_generators.loc[n_full] = cfg.cost_energy[gen_type][inputs.year][inputs.scenario_energy_cost]
            except KeyError:
                pass

        # Curtailed load [MW] -> curtailed energy [MWh]
        energy_curtailed = np.multiply(curtailed_per_unit, 24 / cfg.t_periods)
        self.cost_periods.cost_curtail = energy_curtailed.mul(curt_cost_generators).sum(1)  # Cost per time step
        self.cost.cost_curtail = self.cost_periods.cost_curtail.sum()  # Cost total

        return energy_curtailed.sum().sum()  # [MWh]

    def v2g_cost_losses_battery(self, pevs, inputs, grid, split=False):
        """Calculated the charging cost related to V2G charging from energy losses during the charging process and from
        battery degradation, both due to the additional energy cycled through the battery (compared to charging need)"""

        logger.info('Starting the calculation of charging losses and\n'
                    'battery degradation cost for V2G on process {}'.format(os.getpid()))

        # Unpack pevs.charging_history: [PEV, time step, Charging value]
        charging_history = np.empty((pevs.data.shape[0], cfg.t_periods)).astype('float16')
        charging_history[pevs.charging_history[0], pevs.charging_history[1]] = pevs.charging_history[2]

        # Charging load for each PEV:
        pev_el_load_negative = np.clip(charging_history,
                                       a_min=None, a_max=0).astype('float32')  # Set positive loads to 0 [kW]

        # Calculate capacity losses and their cost and sum across nodes:
        discharge_t = np.zeros(inputs.index.size)
        for t in range(inputs.index.size):
            discharge_t[t] = pev_el_load_negative[pevs.get_pev_select(inputs, grid, t), t % cfg.t_periods].sum()
        self.cost_periods.cost_charge_loss = np.multiply(
            np.multiply(discharge_t,
                        ((24 / cfg.t_periods) *
                         cfg.cost_energy['charging'][inputs.year][inputs.scenario_energy_cost]
                         / -1000)),  # [kW] -> [kWh] -> [EUR]
            1 - (cfg.efficiency ** 2))
        self.cost.cost_charge_loss = self.cost_periods.cost_charge_loss.sum()  # Cost for one day

        energy_discharged = 0
        if inputs.charging_tech == 'V2G':
            # Calculate battery degradation cost:
            # See: Pelzer (2014), cost = (1/(dod_factor * battery capacity)) * cost_batt * energy
            splits = int(grid.pkw_number / 700)  # cfg.splits[inputs.year]
            p1 = pev_el_load_negative.shape[0] / splits
            for run in range(splits if split else 1):
                dc_energy = pev_el_load_negative[int(run * p1):int((run + 1) * p1), :] if splits \
                    else pev_el_load_negative
                dc_energy = pd.DataFrame(dc_energy.T * (24 / cfg.t_periods),
                                         dtype='float32')  # Calc energy from power [kWh]
                dc_energy = dc_energy.loc[:, (dc_energy != 0).any(axis=0)]  # Drop rides that are not discharging
                dc_energy = dc_energy.replace(0, np.NaN)  # Replace non discharging times with NaN

                cumsum = dc_energy.cumsum().fillna(method='pad')  # Calc cumsum and fill NaN times forward
                result = np.array(dc_energy.where(
                    dc_energy.notnull(),
                    -cumsum[dc_energy.isnull()].ffill().diff().fillna(cumsum)[dc_energy.isnull()],  # old reset
                    axis=1).cumsum()).astype('float16')  # Get energy discharged in the next time period

                cycles_energy = np.array(dc_energy.isnull()).astype('bool')  # Get dc_energy with isnull()
                cycles_energy = np.vstack([cycles_energy[1:], ~cycles_energy[47]])
                cycles_energy = np.absolute(np.multiply(result, cycles_energy))
                cycles_energy[cycles_energy <= 0.0000001] = np.NaN  # [kWh] drop rounding errors

                # DOD at time t for vehicle v
                dod_factor = np.divide(cycles_energy,
                                       pevs.data['batt_capacity'][dc_energy.columns]).astype('float32')  # == DOD
                dod_factor = np.multiply(np.power(np.divide(145.71, dod_factor),
                                                  (1 / 0.6844)),
                                         dod_factor).astype('float32')
                dod_factor = np.divide(1, np.multiply(dod_factor,
                                                      pevs.data['batt_capacity'][dc_energy.columns])).astype('float32')
                cost_bat = np.multiply((cfg.cost_infra['battery'][inputs.year][inputs.scenario_energy_cost]),
                                       pevs.data['batt_capacity'][dc_energy.columns]).astype('int')  # MWh -> kWh[EUR]

                # Check if PEV are discharging, if not set cost to 0, if yes calculate battery degradation
                cost_battery_degr = np.nansum(np.multiply(np.multiply(dod_factor, cycles_energy), cost_bat),
                                              axis=1)  # Sum excluding nan values

                self.cost.cost_battery_degr += cost_battery_degr.sum()  # Cost one day
                energy_discharged += dc_energy.sum().sum()  # [kWh]
                logger.info('Finished battery degradation cost calculation run {}/{} on process {} - cost = {}'.format(
                    run + 1, splits, os.getpid(), cost_battery_degr.sum()))

            self.cost_periods.cost_battery_degr = \
                np.multiply(np.clip(pevs.ppev_load, a_min=None, a_max=0).sum(1),
                            np.divide(self.cost.cost_battery_degr,
                                      np.clip(pevs.ppev_load, a_min=None, a_max=0).sum()))
        else:
            self.cost.cost_battery_degr = 0  # No degradation if no discharging is allowed

        return energy_discharged  # [kWh]

    def invest_cost_infra_comms(self, pev, inputs):
        self.investments_pev = cfg.cost_infra[inputs.charging_tech][inputs.year][inputs.scenario_cc_cost] * \
                               pev.number_chargers['total_smart']
        self.cost.annuities_charge += self.investments_pev * cfg.annuity_factor(cost_type='charging')
        self.cost.cost_communication += (cfg.cost_infra['communication'][inputs.year][inputs.scenario_cc_cost] *
                                         pev.number_chargers['total_smart'] if
                                         inputs.charging_tech != 'UCC' or inputs.charging_tech != 'UCC_delayed'
                                         else 0)

        self.cost_periods.annuities_charge = pd.Series(self.cost.annuities_charge / inputs.index.shape[0],
                                                       index=inputs.index)
        self.cost_periods.cost_communication = pd.Series(self.cost.cost_communication / inputs.index.shape[0],
                                                         index=inputs.index)

    def grid_losses(self, grid, inputs, daily):
        apparent_power_losses = pd.Series(
            grd.calc_apparent_power(grid.edisgo.network.results.grid_losses.p.values,  # [kW]
                                    grid.edisgo.network.results.grid_losses.q.values),  # [kvar]
            index=grid.edisgo.network.results.grid_losses.p.index)

        if inputs.selected_steps:
            # Adapt according to weights
            try:
                self.cost_periods.cost_grid_losses = \
                    apparent_power_losses.loc[grid.random_sample.index].div(1000).mul(grid.random_sample.weight).mul(
                        24 / cfg.t_periods).mul(cfg.cost_energy['losses'][inputs.year][inputs.scenario_energy_cost]
                                                )  # div(1000): [kW] -> [MW] -> [MWh]
            except KeyError:  # If timesteps in grid.random_sample could not be calculated in loadflow
                import dill

                file = '{}_{}out_e{}_p{}_c{}'.format(
                    inputs.out_file, 'costerror', inputs.episode, os.getpid(), self.cost.sum())
                for variable, var_name in zip([grid, self], ['grid', 'cost']):
                    try:
                        with open('{}_{}.dill'.format(file, var_name), 'wb') as f:
                            dill.dump(variable, f)
                    except (MemoryError, OSError) as e:
                        logger.warning('Could not save {} due to {}'.format(file, e))

                # Calculate alternative grid losses [kW] -> [MW] -> [MWh]
                try:
                    self.cost_periods.cost_grid_losses = \
                        apparent_power_losses.div(1000).mul(inputs.index.shape[0] / apparent_power_losses.shape[0]
                                                            ).mul(24 / cfg.t_periods).mul(
                            cfg.cost_energy['losses'][inputs.year][inputs.scenario_energy_cost])
                except ZeroDivisionError:
                    self.cost_periods.cost_grid_losses = pd.Series(999999, index=inputs.index[:1])
        else:
            multiplicand = int(inputs.downsample[0]) if inputs.downsample else 24 / cfg.t_periods  # [MW] -> [MWh]
            self.cost_periods.cost_grid_losses = \
                apparent_power_losses.div(1000).mul(multiplicand  # [kW] -> [MW] -> [MWh]
                    ).mul(cfg.cost_energy['losses'][inputs.year][inputs.scenario_energy_cost])
        if daily:
            # Daily individual calculation of cost
            self.cost.cost_grid_losses = self.cost_periods.cost_grid_losses.resample('D', how='sum').dropna()
        else:
            self.cost.cost_grid_losses = self.cost_periods.cost_grid_losses.resample('D', how='sum').dropna().mul(
                inputs.dates_represented if inputs.dates_represented is not None else 1).sum()  # Cost whole year y

    def total(self):
        return self.cost.sum().sum()

    def calc_all_cost(self, grid, pev, inputs, name, curtailed_per_unit=None, daily=False, combined_expansion=False):
        # Charging infrastructure investment annuities & cost
        if inputs.charging_tech != 'UCC':
            self.invest_cost_infra_comms(pev, inputs)

        # Calculate curtailment cost and battery degradation cost
        energy_curtailed = self.curtailment_cost(grid, inputs, curtailed_per_unit) if inputs.curtailment else 0
        energy_discharged = self.v2g_cost_losses_battery(pev, inputs, grid, split=True) if \
            inputs.charging_tech == 'V2G' else 0

        # Invest. annuities
        if not combined_expansion or name == 'base_ucc':
            grid = self.investment_needs(grid, pev.ppev_load, inputs)
            self.grid_losses(grid, inputs, daily=daily)  # Calculate distribution system energy loss cost

        return energy_discharged, energy_curtailed, grid  # [kWh]
