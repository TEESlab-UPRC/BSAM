"""
This module handles unit commitment and presents a MIP alternative to the epl solver
"""

import copy
import numpy
import pandas
import class_library
import gurobipy
import pulp

class UnitCommit:
    """
    This is the Unit Commit class. It is composed of all needed functions to implement the UC algorithm.
    """
    def __init__(self, reserve_margins, backup_reserves, generator_long_term_availability, uc_verbosity):
        """
        Initialize the class, using empty variables where possible
        """
        self.reserve_requirements = reserve_margins
        self.backup_reserves = backup_reserves
        self.generator_long_term_availability = generator_long_term_availability
        # this controls how much the uc module will talk to stdout. possible values range -1 ~ 3, from silent to very verbose
        self.verbosity = uc_verbosity
        # all attributes below are empty initially, but will be updated later on
        # filled with update_self_data
        self.day_demand = "placeholder"
        self.single_plant_agents = "placeholder"
        self.generators = "placeholder"
        self.virtual_generators = "placeholder"
        self.all_generators = "placeholder"
        self.all_generators_by_name = "placeholder"

        # if, due to demand being very low or too many active plants during the last day,
        # the problem is unsolvable while also respecting their MUT, record the excess power here
        # and, if RES power is the reason fo such low demand, reduce that res power so that the problem is solvable
        # the reduction will be done after solving and a failure to find RES there will result to a crash
        self.excess_res_power = "placeholder"
        self.periods = "placeholder"

    def get_available_generators(self, agents, day):
        """
        populates a list of generators available to use in the UC solution
        """
        generators = [agent.plant for agent in agents if \
                        (agent.plant.name not in self.generator_long_term_availability.columns) or (self.generator_long_term_availability.loc[day.year,agent.plant.name] == 1)]
        return generators

    def calculate_demand_margins(self):
        """
        Calculates the required demand + reserve upper & lower margins
        """
        self.day_demand.loc[:,"lower_reserves_margin"] = self.day_demand.loc[:,"demand"] - self.reserve_requirements.loc[:,'FCR_downwards_min'] - self.reserve_requirements.loc[:,'FRR_downwards_min']
        self.day_demand.loc[:,"upper_reserves_margin"] = self.day_demand.loc[:,"demand"] + self.reserve_requirements.loc[:,'FCR_upwards_min'] + self.reserve_requirements.loc[:,'FRR_upwards_min']

    def remove_virtual_generators_from_generator_list(self):
        """
        Moves all virtual generators from self.generators to self.virtual_generators
        Resets self.virtual_generators so be sure to only run this once per day
        """
        virtual_generator_indices = []
        for index,plant in enumerate(self.generators):
            if plant.kind == 'virtual':
                virtual_generator_indices.append(index)
        self.virtual_generators = [self.generators.pop(index) for index in virtual_generator_indices]

    def rank_generators_by_cost_metric(self):
        """
        Sort the generators by their cost_metric inplace
        Rank "must run" plants first & virtual plants last
        """
        self.generators = sorted(self.generators, key=lambda plant: (-plant.must_run, plant.kind in ['virtual'], plant.cost_metric_profit))

    def calculate_power_cost(self):
        """
        Return a dataframe where the cost of power from each plant for the day is saved
        This will be used for dispatch
        """
        # get the segments from each plant and concatenate to a large dataframe
        generator_segments = []
        for plant in self.all_generators:
            generator_segments.append(plant.marginal_costs_by_profit_daily)
        power_cost = pandas.concat(generator_segments)
        # sort costs ascending
        power_cost.sort_values('mw_unit_cost',inplace=True,ascending=True)
        # fix the df index
        power_cost.reset_index(drop=True,inplace=True)
        # and return
        return power_cost

    def index_all_generators_by_name(self):
        """
        Return a dictionary with all the generator names and objects, so they can be refered to by name
        """
        generators_dict = {}
        for plant in self.all_generators:
            generators_dict.update({plant.name:plant})
        return generators_dict

    def update_uc_data(self,day):
        """
        This updates all pertinent data (e.g. demand, agent actions, etc)
        & enforces mut/mdt restrictions due to last day operation as needed.
        It is supposed to run at the start of each day.
        """
        for agent in self.single_plant_agents:
            # update the cost metrics for all plants
            agent.plant.cost_metric_profit = agent.plant.calculate_cost_metric('income')
            # update the real cost for all plants
            agent.plant.marginal_costs_by_profit_daily = agent.plant.calculate_mc_segments('income')
            # enforce required startups/shutdowns due to last day data
            agent.plant.enforce_mut_mdt_restrictions_from_last_day_operation()
            # enforce must_run restrictions
            agent.plant.enforce_must_run_restriction()
        # and update the generator list
        self.generators = self.get_available_generators(self.single_plant_agents,day)
        # rank generators by cost metric, so that they can be allocated to an initial commitment
        self.rank_generators_by_cost_metric()
        # remove any virtual generators from the generator list and keep them separate
        self.remove_virtual_generators_from_generator_list()
        # keep all generators in a specific list
        self.all_generators = self.generators + self.virtual_generators
        # also save all generator objects in a dictionary so that they can be refered to by name
        self.all_generators_by_name = self.index_all_generators_by_name()
        # demand_range is list of tuples [(demand, required_pmin, required_pmax), ....] based on the demand of the day
        self.calculate_demand_margins()
        # update the day periods from demand
        self.periods = self.day_demand.index.size
        self.timeperiods = numpy.arange(self.periods)
        # reset excess_res_power
        self.excess_res_power = pandas.Series(0,index=numpy.arange(self.periods))
        # update & process data for mip
        self.add_first_zero_segment() # insert a segment to all generators, from 0 to pmin, with 0 cost, because units are under Pmin at that point
        self.convert_data_to_integers()

        # populate the dataframe with the cost of power
        self.power_cost = self.calculate_power_cost()

    def add_first_zero_segment(self):
        for agent in self.single_plant_agents:
            agent.plant.marginal_costs_milp = pandas.concat([pandas.DataFrame(0,index=[-1],columns=agent.plant.marginal_costs.columns), agent.plant.marginal_costs], ignore_index=True)
            agent.plant.marginal_costs_milp.loc[0] = [0,0,agent.plant.available_pmin-0.001,agent.plant.name]

    def convert_data_to_integers(self):
        """
        This convert data to integers to reduce solution complexity.
        MW/MWh are converted to KW/KWh to keep ok granularity
        """
        for agent in self.single_plant_agents:
            # convert MW to KW so that generation can be an integer (to reduce solution complexity)
            agent.plant.available_power = round(agent.plant.available_power * 1000).astype(int)
            agent.plant.available_min_power = round(agent.plant.available_min_power * 1000).astype(int)
            agent.plant.marginal_costs_milp.loc[:,'mw_unit_cost'] = agent.plant.marginal_costs_milp.loc[:,'mw_unit_cost'] / 1000
            agent.plant.marginal_costs_milp.loc[:,['segment_pmin','segment_pmax']] = agent.plant.marginal_costs_milp.loc[:,['segment_pmin','segment_pmax']] * 1000
        # convert MW to KW so that demand & reserves can be an int (to reduce complexity)
        self.day_demand = round(self.day_demand * 1000).astype(int)

    def convert_data_to_mw(self):
        """
        This convert data to mw as it should be for wmsim
        kW/kWh are converted to MW/MWh
        """
        for agent in self.single_plant_agents:
            # convert MW to KW so that generation can be an integer (to reduce solution complexity)
            agent.plant.available_power = agent.plant.available_power / 1000.0
            agent.plant.available_min_power = agent.plant.available_min_power / 1000.0
            agent.plant.marginal_costs_milp.loc[:,'mw_unit_cost'] = agent.plant.marginal_costs_milp.loc[:,'mw_unit_cost'] * 1000
            agent.plant.marginal_costs_milp.loc[:,['segment_pmin','segment_pmax']] = agent.plant.marginal_costs_milp.loc[:,['segment_pmin','segment_pmax']] / 1000.0
            agent.plant.online_data.loc[:,'power'] = agent.plant.online_data.loc[:,'power'] / 1000.0
        # convert MW to KW so that demand & reserves can be an int (to reduce complexity)
        self.day_demand = self.day_demand / 1000.0

    def create_uc_model(self):
        # create model
        self.uc_model = pulp.LpProblem("the_daily_uc_problem", pulp.LpMinimize)
        uc_model_variables = self.create_uc_model_variables()
        # now start populating the model
        # the objective function equals the total system cost
        self.set_total_system_cost_objective(uc_model_variables)
        # constraints
        # generation must match demand in each period
        self.set_generation_demand_match_constr(uc_model_variables)
        # unit status must be 0 if generation is under pmin, 1 otherwise & generation must be 0 if it is under pmin
        self.set_unit_status_constraints(uc_model_variables)
        # a cost segment is active only if generation is within its boundaries
        self.set_unit_cost_constraints(uc_model_variables)
        # the per-energy-unit cost of each period must be the cost of the most expensive online unit in the period
        self.set_system_cost_constraints(uc_model_variables)
        # units must always respect MUT & MDT constraints - TODO: Allow relaxation to this constr (implement as a soft constraint with a large cost)
        #self.set_unit_uptime_downtime_constraints(uc_model_variables)
        self.set_sr_constraints(uc_model_variables)
        self.uc_model_variables = uc_model_variables

    def create_uc_model_variables(self):
        # create a dict to put every primary variable concerning units inside - initialize with false
        variable_names = ['status','mode','generation','startup','shutdown','unit_cost_segments_activity','unit_gen_over_segment_pmin','unit_gen_under_segment_pmax','spinning_reserve_up','spinning_reserve_down']

        uc_model_variables = {unit_name:{variable:False for variable in variable_names} for unit_name,unit in self.all_generators_by_name.items()}
        # create the uc_model_variables for each plant and period and add them to the uc_model_variables dict
        for unit_name,unit in self.all_generators_by_name.items():
            # first create the vars for generators
            status = pulp.LpVariable.dicts('status_'+unit_name,indexs=self.timeperiods,cat='Binary')
            #generation = pulp.LpVariable.dicts("generation_"+unit_name,indexs=self.timeperiods,lowBound=0,upBound=unit.available_pmax,cat='Integer')
            generation = pulp.LpVariable.dicts("generation_"+unit_name,indexs=self.timeperiods,lowBound=0,cat='Integer')
            startup = pulp.LpVariable.dicts(name='startup_'+unit_name,indexs=self.timeperiods,cat='Binary')
            shutdown = pulp.LpVariable.dicts(name='shutdown_'+unit_name,indexs=self.timeperiods,cat='Binary')
            unit_cost = pulp.LpVariable.dicts(name='unit_cost_'+unit_name,indexs=self.timeperiods,cat='Integer')
            unit_spinning_reserve_up = pulp.LpVariable.dicts(name='SRup_'+unit_name,indexs=self.timeperiods,cat='Integer')
            unit_spinning_reserve_down = pulp.LpVariable.dicts(name='SRdown_'+unit_name,indexs=self.timeperiods,cat='Integer')
            unit_provides_spinning_reserve_up = pulp.LpVariable.dicts(name='provides_spinning_reserve_up_'+unit_name,indexs=self.timeperiods,cat='Binary')
            unit_provides_spinning_reserve_down = pulp.LpVariable.dicts(name='provides_spinning_reserve_down_'+unit_name,indexs=self.timeperiods,cat='Binary')

            unit_cost_segments_activity = []
            unit_gen_over_segment_pmin = []
            unit_gen_under_segment_pmax = []
            for segment_index in unit.marginal_costs.index:
                unit_cost_segments_activity.append(pulp.LpVariable.dicts('cost_segment_activity_'+str(segment_index)+'_'+unit_name,indexs=self.timeperiods,cat='Binary'))
                unit_gen_over_segment_pmin.append(pulp.LpVariable.dicts('unit_gen_over_segment_pmin_'+str(segment_index)+'_'+unit_name,indexs=self.timeperiods,cat='Binary'))
                unit_gen_under_segment_pmax.append(pulp.LpVariable.dicts('unit_gen_under_segment_pmax_'+str(segment_index)+'_'+unit_name,indexs=self.timeperiods,cat='Binary'))

            # then add them to the dict
            uc_model_variables[unit_name]['status'] = status
            uc_model_variables[unit_name]['generation'] = generation
            uc_model_variables[unit_name]['startup'] = startup
            uc_model_variables[unit_name]['shutdown'] = shutdown
            uc_model_variables[unit_name]['SRup'] = unit_spinning_reserve_up
            uc_model_variables[unit_name]['SRdown'] = unit_spinning_reserve_down
            uc_model_variables[unit_name]['provides_SRup'] = unit_provides_spinning_reserve_up
            uc_model_variables[unit_name]['provides_SRdown'] = unit_provides_spinning_reserve_down
            uc_model_variables[unit_name]['unit_cost_segments_activity'] = unit_cost_segments_activity
            uc_model_variables[unit_name]['unit_gen_over_segment_pmin'] = unit_gen_over_segment_pmin
            uc_model_variables[unit_name]['unit_gen_under_segment_pmax'] = unit_gen_under_segment_pmax

        # also add to the list the uc_model_variables for the cost of each period
        uc_model_variables['period_generation_cost'] = pulp.LpVariable.dicts(name='period_generation_cost',indexs=self.timeperiods,lowBound=0,cat='Continuous')
        return uc_model_variables

    def set_total_system_cost_objective(self,uc_model_variables):
        # get the system costs per each product type
        system_generation_cost = [uc_model_variables['period_generation_cost'][period]*self.day_demand.loc[period,'demand'] for period in self.timeperiods]
        system_startup_cost = [uc_model_variables[unit_name]['startup'][period]*unit.start_costs[0] for unit_name,unit in self.all_generators_by_name.items() for period in self.timeperiods]
        # the total_system_cost is just the sum of the costs
        total_system_cost = sum(system_generation_cost)+sum(system_startup_cost)
        # add the obj finction to the model
        self.uc_model += total_system_cost, 'Total System Cost'

    def set_generation_demand_match_constr(self,uc_model_variables):
        # generation must always match demand
        for period in self.timeperiods:
            self.uc_model += sum(uc_model_variables[unit_name]['generation'][period] for unit_name,unit in self.all_generators_by_name.items()) == self.day_demand.loc[period,'demand'], "Constr_system_generation_period_%s"%str(period)

    def set_unit_status_constraints(self,uc_model_variables):
        # constraints for correlation between unit generation / pmin / status
        for period in self.timeperiods:
            for unit_name,unit in self.all_generators_by_name.items():
                # extract unit pmin for readability
                unit_pmax = unit.available_power[period]
                unit_pmin = unit.available_min_power[period]
                # status of each unit must be 1 if generation >= Pmin and 0 otherwise - this is the same with below?
                # probably s >= (g - pmin + 1) / pmax is best

                #self.uc_model += uc_model_variables[unit_name]['status'][period] <= 1 + (uc_model_variables[unit_name]['generation'][period]-unit_pmin)/unit_pmax, "Constr_status_period_%s_unit_%s"%(str(period), unit_name)
                self.uc_model += uc_model_variables[unit_name]['status'][period] >= (uc_model_variables[unit_name]['generation'][period]-unit_pmin)/unit_pmax, "Constr_status2_period_%s_unit_%s"%(str(period), unit_name)

                # generation of each unit must be >= pmin if status = 1 and 0 otherwise
                self.uc_model += uc_model_variables[unit_name]['status'][period]*unit_pmin <= uc_model_variables[unit_name]['generation'][period], "Constr_pmin_period_%s_unit_%s"%(str(period), unit_name)
                # generation must be under pmax
                self.uc_model += uc_model_variables[unit_name]['generation'][period] <= unit_pmax, "Constr_pmax_period_%s_unit_%s"%(str(period), unit_name)

    def set_unit_cost_constraints(self,uc_model_variables):
        # constraints for correlation between unit generation / segment activity / cost
        for period in self.timeperiods:
            for unit_name,unit in self.all_generators_by_name.items():
                # sum of segments for a unit within a period must be equal to its status
                self.uc_model += uc_model_variables[unit_name]['status'][period] == sum(uc_model_variables[unit_name]['unit_cost_segments_activity'][segment_index][period] for segment_index in unit.marginal_costs.index), "Constr_segment_sum_period_%s_unit_%s"%(str(period), unit_name)
                for segment_index in range(len(unit.marginal_costs)):
                    # get the segment cost and boundaries (the 1st segment need not be checked, if all other segments equal 0, that segment would be 1)
                    if segment_index != 0:
                        segment_pmin = unit.marginal_costs.loc[segment_index,'segment_pmin']
                        segment_pmax = unit.marginal_costs.loc[segment_index,'segment_pmax']
                        unit_pmax = unit.available_power[period]

                        # only the linear segment of the cost function where plant generation is within its bounds must be active, the rest must be inactive
                        # since this implies a nonlinear function, we use instead two linear functions to express this
                        # unit_gen_over_segment_pmin is 1 if unit generation is over segment Pmin, and 0 otherwise. So its ocnstraints are:
                        # unit_gen_over_segment_pmin >= (generation - segmentPmin + 0.5) / unitPmax
                        # 1-unit_gen_over_segment_pmin >= (segmentPmin - generation - 0.5) / unitPmax
                        # the 0.5 is used to offset the constrain so that the output will be clear when on a boundary. Since unit generation is an integer, 0.5 will never offset too much
                        self.uc_model += uc_model_variables[unit_name]['unit_gen_over_segment_pmin'][segment_index][period] >= (uc_model_variables[unit_name]['generation'][period] - segment_pmin + 0.5) / unit_pmax, "Constr_unit_gen_over_segment_pmin_activity_period_%s_unit_%s_segment_%s"%(str(period), unit_name, str(segment_index))
                        self.uc_model += 1 - uc_model_variables[unit_name]['unit_gen_over_segment_pmin'][segment_index][period] >= (segment_pmin - uc_model_variables[unit_name]['generation'][period] - 0.5) / unit_pmax, "Constr_unit_gen_over_segment_pmin_inactivity_period_%s_unit_%s_segment_%s"%(str(period), unit_name, str(segment_index))
                        # unit_gen_under_segment_pmax is 1 if unit generation is under segment Pmax, and 0 otherwise
                        # the constrains follow a similar logic as with above
                        self.uc_model += uc_model_variables[unit_name]['unit_gen_under_segment_pmax'][segment_index][period] >= (segment_pmax - uc_model_variables[unit_name]['generation'][period] + 0.5) / unit_pmax, "Constr_unit_gen_under_segment_pmax_activity_period_%s_unit_%s_segment_%s"%(str(period), unit_name, str(segment_index))
                        self.uc_model += 1 - uc_model_variables[unit_name]['unit_gen_under_segment_pmax'][segment_index][period] >= (uc_model_variables[unit_name]['generation'][period] - segment_pmax - 0.5) / unit_pmax, "Constr_unit_gen_under_segment_pmax_inactivity_period_%s_unit_%s_segment_%s"%(str(period), unit_name, str(segment_index))
                        # now, segment activity must be equal to unit_gen_over_segment_pmin + unit_gen_under_segment_pmax - 1
                        self.uc_model += uc_model_variables[unit_name]['unit_cost_segments_activity'][segment_index][period] == uc_model_variables[unit_name]['unit_gen_under_segment_pmax'][segment_index][period] + uc_model_variables[unit_name]['unit_gen_over_segment_pmin'][segment_index][period] - 1, "Constr_segment_activity_period_%s_unit_%s_segment_%s"%(str(period), unit_name, str(segment_index))

    def set_system_cost_constraints(self,uc_model_variables):
        # constraints for the per MW cost of each period
        # the per MW cost of each period is the cost of the most expensive unit, it suffices that period_cost >= segment_activity*segment_cost for each period, generator, and segment
        for period in self.timeperiods:
            for unit_name,unit in self.all_generators_by_name.items():
                for segment_index in unit.marginal_costs.index:
                    self.uc_model += uc_model_variables['period_generation_cost'] >= uc_model_variables[unit_name]['unit_cost_segments_activity'][segment_index][period]*unit.marginal_costs.loc[segment_index,'mw_unit_cost'], "Constr_min_period_cost_period_%s_unit_%s_segment_%s"%(str(period), unit_name, str(segment_index))

    def set_unit_uptime_downtime_constraints(self,uc_model_variables):
        # constraints for startup/shutdown / uptime / downtime
        for unit_name,unit in self.all_generators_by_name.items():
            mut_p = unit.min_uptime
            mdt_p = unit.min_downtime
            ut_init_p = unit.online_data.loc[-1,'uptime']
            dt_init_p = unit.online_data.loc[-1,'downtime']
            for period in self.timeperiods:
                # get the value of the last period first
                if period == 0:
                    status_minus_1_p = int(bool(ut_init_p))
                else:
                    status_minus_1_p = uc_model_variables[unit_name]['status'][period-1]

                self.uc_model += uc_model_variables[unit_name]['startup'][period] >= uc_model_variables[unit_name]['status'][period]-status_minus_1_p, "Constr_startup_activity_period_start_%s_unit_%s"%(str(period), unit_name)
                self.uc_model += uc_model_variables[unit_name]['startup'][period] <= uc_model_variables[unit_name]['status'][period]-status_minus_1_p+1, "Constr_startup_activity_period_end_%s_unit_%s"%(str(period), unit_name)
                self.uc_model += -uc_model_variables[unit_name]['startup'][period] <= uc_model_variables[unit_name]['status'][period]-status_minus_1_p, "Constr_startup_activity_period_zero_%s_unit_%s"%(str(period), unit_name)

                self.uc_model += uc_model_variables[unit_name]['shutdown'][period] >= status_minus_1_p-uc_model_variables[unit_name]['status'][period], "Constr_shutdown_activity_period_start_%s_unit_%s"%(str(period), unit_name)
                self.uc_model += uc_model_variables[unit_name]['shutdown'][period] <= status_minus_1_p-uc_model_variables[unit_name]['status'][period]+1, "Constr_shutdown_activity_period_end_%s_unit_%s"%(str(period), unit_name)
                self.uc_model += -uc_model_variables[unit_name]['shutdown'][period] <= status_minus_1_p-uc_model_variables[unit_name]['status'][period], "Constr_shutdown_activity_period_zero_%s_unit_%s"%(str(period), unit_name)

                # when we are starting up, observe min uptime
                startups_within_mut = []
                if period+1-mut_p >= 0:
                    startups_within_mut = [uc_model_variables[unit_name]['startup'][p] for p in range(period+1-mut_p, period+1)]
                elif ut_init_p != 0:
                    startups_within_mut = [uc_model_variables[unit_name]['startup'][p] for p in range(0, period+1)]
                    if abs(period+1-mut_p) >= ut_init_p:
                        startups_within_mut.append(1)
                self.uc_model += uc_model_variables[unit_name]['status'][period] >= sum(startups_within_mut), "Constr_min_mut_period_%s_unit_%s"%(str(period), unit_name)

                # when we are shutting down, observe min downtime
                shutdowns_within_mdt = []
                if period+1-mdt_p >= 0:
                    shutdowns_within_mdt = [uc_model_variables[unit_name]['shutdown'][p] for p in range(period+1-mdt_p, period+1)]
                elif dt_init_p != 0:
                    shutdowns_within_mdt = [uc_model_variables[unit_name]['shutdown'][p] for p in range(0, period+1)]
                    if abs(period+1-mdt_p) >= dt_init_p:
                        shutdowns_within_mdt.append(1)
                self.uc_model += 1-uc_model_variables[unit_name]['status'][period] >= sum(shutdowns_within_mdt), "Constr_min_mdt_period_%s_unit_%s"%(str(period), unit_name)

    # constraints for reserves
    def set_sr_constraints(self,uc_model_variables):
        # if no requirement for a reserve exists, its cost is set to be 0, else constraints are set to procure it and calculate its cost
        # SRup & SRdown
        # the available SR to the system is the aggregated SR of all committed units, so long as their available unused capacity is enough. If not, the SR they contribute is lowered as needed
        for period in self.timeperiods:
            system_SRup = self.day_demand.loc[period,'upper_reserves_margin'] - self.day_demand.loc[period,'demand']
            system_SRdown = self.day_demand.loc[period,'demand'] - self.day_demand.loc[period,'lower_reserves_margin']
            if system_SRup > 0:
                self.uc_model += sum(uc_model_variables[unit_name]['SRup'][period] for unit_name,unit in self.all_generators_by_name.items()) >= system_SRup, "Constr_system_SRup_%s"%(str(period))
            if system_SRdown > 0:
                self.uc_model += sum(uc_model_variables[unit_name]['SRdown'][period] for unit_name,unit in self.all_generators_by_name.items()) >= system_SRdown, "Constr_system_SRdown_%s"%(str(period))

            for unit_name,unit in self.all_generators_by_name.items():
                unit_status = uc_model_variables[unit_name]['status'][period]
                # SRup <= pmax - generation
                self.uc_model += uc_model_variables[unit_name]['SRup'][period] <= unit.available_power[period]-uc_model_variables[unit_name]['generation'][period], "Constr_SRup_generation_%s_unit_%s"%(str(period), unit_name)
                # SRup <= (pmax-pmin)*status
                self.uc_model += uc_model_variables[unit_name]['SRup'][period] <= (unit.available_power[period]-unit.available_min_power[period])*unit_status, "Constr_SRup_maxcap_%s_unit_%s"%(str(period), unit_name)

                # SRdown <= generation - pmin
                #self.uc_model += uc_model_variables[unit_name]['SRdown'][period] <= uc_model_variables[unit_name]['generation'][period]-unit.available_min_power[period], "Constr_SRdown_generation_%s_unit_%s"%(str(period), unit_name)

                # SRdown <= (pmax-pmin)*status
                self.uc_model += uc_model_variables[unit_name]['SRdown'][period] <= (unit.available_power[period]-unit.available_min_power[period])*unit_status, "Constr_SRdown_maxcap_%s_unit_%s"%(str(period), unit_name)

                # activity and cost calculation constraints are used only if there is an actual need to procure a reserve type - else cost=0 and no constraints are set
                # a unit provides reserves if provides_reserve <= unit_reserve AND provides_reserve >= unit_reserve*(1/system_reserve_demand)
                # a period's reserves cost is always equal to the cost of the most expensive plant that provides this reserve
                if system_SRup > 0:
                    self.uc_model += uc_model_variables[unit_name]['provides_SRup'][period] <= uc_model_variables[unit_name]['SRup'][period], "Constr_SRup_activity_high_%s_unit_%s"%(str(period), unit_name)
                    self.uc_model += uc_model_variables[unit_name]['provides_SRup'][period] >= uc_model_variables[unit_name]['SRup'][period]*(1/system_SRup), "Constr_SRup_activity_low_%s_unit_%s"%(str(period), unit_name)
                if system_SRdown > 0:
                    self.uc_model += uc_model_variables[unit_name]['provides_SRdown'][period] <= uc_model_variables[unit_name]['SRdown'][period], "Constr_SRdown_activity_high_%s_unit_%s"%(str(period), unit_name)
                    self.uc_model += uc_model_variables[unit_name]['provides_SRdown'][period] >= uc_model_variables[unit_name]['SRdown'][period]*(1/system_SRdown), "Constr_SRdown_activity_low_%s_unit_%s"%(str(period), unit_name)

    def check_for_and_do_res_curtailment(self):
        """
        Makes sure the required demand & reserves can be met.
        If not, the only explanation is very high RES.
        In that case there should be curtailment or redispatching
        """
        # create an initial commitment
        current_commitment = self.create_commitment_from_generator_data()
        min_reserves_restriction = current_commitment.commitment_power.pmin - self.day_demand.lower_reserves_margin
        if (min_reserves_restriction > 0).any():
            # this would mean that we cannot satisfy correct operation as generated power minimum  exceeds reserves lower margin!
            # keep the generators online and increase demand instead as needed, also recording the increase
            # this is done to allow energy market to reduce res production so that the problem to be solvable
            # the reduction is done post-uc-solution
            print ("WARNING! Required generated power exceeds reserves lower margin")
            print('now fixing demand and will attempt to reduce RES power after UC solve to compensate')
            # Find how much energy must be curtailed. This shall equal the positive values of min_reserves_restriction
            min_reserves_restriction.loc[min_reserves_restriction < 0] = 0
            # increase demand & reserves limits accordingly
            self.day_demand.loc[:,'demand'] += min_reserves_restriction
            self.day_demand.loc[:,'lower_reserves_margin'] += min_reserves_restriction
            self.day_demand.loc[:,'upper_reserves_margin'] += min_reserves_restriction
            # save the excess energy values for lated parse
            self.excess_res_power = min_reserves_restriction.values

    def update_agent_plants_with_model_results(self):
        """
        This updates the generator data depending on the solution found for the model
        The plants "online data" dataframe is updated
        """
        for unit_name,unit in self.all_generators_by_name.items():
            power_list = []
            for period in self.day_demand.index:
                status = self.uc_model_variables[unit_name]['status'][period].varValue
                # update online status
                unit.online_data.loc[period,'online'] = round(status)
                # get the power into a list & convert back to MW/MWh
                power_list.append(round(self.uc_model_variables[unit_name]['generation'][period].varValue))
            # now update everything bar the power
            unit.update_uptime_downtime_data()
            # and the power after that
            unit.online_data.loc[0:23,'power'] = power_list

    def test_solution_results(self):
        """
        This is used to verify all's well within the mip solution
        checks ut,dt, transitions & statuses, also demand and reserves
        """
        online_datas = []
        for unit_name,unit in self.all_generators_by_name.items():
            if unit.find_mut_problems().any():
                print ('found MUT problems')
                import ipdb;ipdb.set_trace()
            if unit.find_mdt_problems().any():
                print('found MDT problems')
                import ipdb;ipdb.set_trace()

            online_power = unit.online_data.loc[0:23,'power'].to_frame()
            pmins = []
            pmaxs = []
            for period in (self.day_demand.index):
                if unit.online_data.loc[period,'power'] > 0 or unit.available_min_power.loc[period] == 0:
                    pmins.append(unit.available_min_power.loc[period])
                    pmaxs.append(unit.available_power.loc[period])
                else:
                    pmins.append(0)
                    pmaxs.append(0)
                    if unit.online_data.loc[period,'power'] > 0:
                        print('plant offline but power is generated')
                        import ipdb;ipdb.set_trace()
            online_power.loc[0:23,'pmin'] = pmins
            online_power.loc[0:23,'pmax'] = pmaxs
            online_datas.append(online_power)

        total_generation_pmin = pandas.concat(online_datas,axis=1).loc[0:23,'pmin'].sum(1)
        total_generation_pmax = pandas.concat(online_datas,axis=1).loc[0:23,'pmax'].sum(1)
        total_generation = pandas.concat(online_datas,axis=1).loc[0:23,'power'].sum(1)

        for period in range(24):
            if not numpy.isclose(self.day_demand.loc[period,'demand'],total_generation.loc[period]):
                print ('demand not satisfied')
                import ipdb;ipdb.set_trace()
            if self.day_demand.loc[period,'lower_reserves_margin'] < total_generation_pmin.loc[period]:
                print ('demand lower bound not satisfied')
                import ipdb;ipdb.set_trace()
            if self.day_demand.loc[period,'upper_reserves_margin'] > total_generation_pmax.loc[period]:
                print ('demand upper bound not satisfied')
                print ('missing demand',self.day_demand.loc[period,'upper_reserves_margin']-total_generation_pmax.loc[period])
                import ipdb;ipdb.set_trace()
        # if all's well, print OK
        print ('RESULT TESTS OK!!!')

    def get_solution_final_cost(self):
        """
        Returns the real cost of the solution found.
        Iterates over all generators and find the most expensive one for each hour, this cost being the smp
        """
        daily_solution_cost = 0
        daily_smp = []
        for hour in range(24):
            hourly_smp = 0
            for plant in self.generators:
                hourly_smp = max(hourly_smp,plant.calculate_hourly_smp(hour))
            daily_solution_cost += self.day_demand[hour][0] * hourly_smp
            daily_smp.append(hourly_smp)
        return daily_solution_cost,daily_smp

    def do_economic_dispatch(self, commitment):
        """
        Do an economic dispatch on the commitment given, apply it to the generators and find the marginal price (price to get 1 more unit of power)

        This is the most expensive function of the module (and one of the two worst in wmsim).
        It should be possible to optimize its performance by unpacking the plant objects into arrays and then writing the main loop in C
        """
        # set generation of all online plants to pmin and for each period and online plant get all power/cost segments
        for plant in self.all_generators:
            # initial dispatch power is status * available_min_power
            plant.online_data.loc[0:23,'power'] = commitment.commitment_values.loc[:,plant.name] * plant.available_min_power.loc[commitment.commitment_values.index]

        smp = pandas.Series(index=self.day_demand.index,name='smp',dtype='float64')
        # for each period add more power to the given commitment, mw by mw until demand is satisfied
        for period in self.day_demand.index:
            commited_power = commitment.commitment_power.loc[period,'pmin']
            for power_cost_index in self.power_cost.index:
                # get the required power to meet demand
                period_generation_delta = self.day_demand.loc[period,'demand'] - commited_power

                # also get the next cheapest generator segment - it is the first on the power_cost list
                plant = self.all_generators_by_name[self.power_cost.loc[power_cost_index,'generator']]
                plant_generation_delta = 0
                if commitment.commitment_values.loc[period,plant.name] > 0:
                    plant_generation_delta = self.power_cost.loc[power_cost_index,'segment_pmax'] - plant.online_data.loc[period,'power']

                # if that plant is not already generating more power than the segment offers
                if plant_generation_delta > 0:
                    # if the offered power is less than what is required to meet demand
                    if period_generation_delta > plant_generation_delta:
                        # dispatch the plant in this position
                        plant.online_data.loc[period,'power'] = self.power_cost.loc[power_cost_index,'segment_pmax']
                        # update the commited_power and try the next index for more power
                        commited_power += plant_generation_delta
                        # save this plant's smp unless it is a virtual plant - this will be overwritten during the next loop unless a virtual plant is used
                        if plant.kind != 'virtual':
                            smp.loc[period] = self.power_cost.loc[power_cost_index,'mw_unit_cost']
                    # if the plant can generate equal or more than needed to meet demand
                    else:
                        # dispatch the plant up to demand
                        plant.online_data.loc[period,'power'] += period_generation_delta
                        # update the commited_power and try the next index for more power
                        commited_power += period_generation_delta
                        # this plant's cost is the smp unless it is a virtual plant - in that case smp is the smp of the last non-virtual plant
                        if plant.kind != 'virtual':
                            smp.loc[period] = self.power_cost.loc[power_cost_index,'mw_unit_cost']
                        # finally break out of the loop as the period is solved
                        break
        # now that the dispatch was completed, also get the real cost of the system
        # this cost will not equal smp * demand because (1) startup costs were not taken into account and (2) the problem is non-convex
        # at such an instance, all commited generators with energy unit cost higher than the smp will be remunerated at their bid
        final_cost = self.calculate_final_cost(smp)

        # and return the data
        return smp,final_cost


    def calculate_final_cost(self,smp):
        """
        Calculates the final cost of the dispatch using a single core/thread
        """
        final_cost = 0
        for plant in self.all_generators:
            final_cost += plant.calculate_one_day_income(smp)
        return final_cost

    def create_commitment_from_generator_data(self):
        """
        Returns a plant commitment as per all generators' online_data
        """
        # create a new commitment
        new_commitment = class_library.Plant_Commitment(self.all_generators, self.day_demand.index)
        for period in self.day_demand.index:
            for plant in self.all_generators:
                if plant.online_data.loc[period,"online"] == 1:
                    new_commitment.add_plant([period],plant)
        return new_commitment

    def calculate_unit_commit(self,day):
        """
        This is the main function of the class. Does what it says on the tin.
        """
        if self.verbosity > -1:
            print ('start new unit commit calculation')
        # step 0 - update the uc data to get ready for running the algorithm
        self.update_uc_data(day)
        # Step 1 - create a milp model with the problem
        self.create_uc_model()
        # Step 2 - solve the model
        uc_status = "not_modelled"
        # if self.config_options.solver == "cbc":
            # uc_status = self.uc_model.solve(pulp.COIN_CMD(msg=True,threads=4))
        # elif self.config_options.solver == "gurobi":
            # uc_status = self.uc_model.solve(pulp.GUROBI_CMD(msg = 1)) # last test xx seconds
        #uc_status = self.uc_model.solve()
        #uc_status2 = self.uc_model.solve(pulp.COIN_CMD(msg=True))
        uc_status3 = self.uc_model.solve(pulp.GUROBI(msg=1))
        #uc_status4 = self.uc_model.solve(pulp.GUROBI_CMD())
        #print('s1',uc_status)
        #print('s2',uc_status2)
        print('s3',uc_status3)
        #print('s4',uc_status4)
        # for v in self.uc_model.variables():
            # if 'status' in v.name:
                # print(v.name, "=", v.varValue)
        # step 3 - extract the results from the problem and update the generators as needed
        self.update_agent_plants_with_model_results()

        #import ipdb;ipdb.set_trace()
        # re-scale generation & demand to MW/MWh
        self.convert_data_to_mw()

        # also test the solution before going on
        self.test_solution_results()
        # step 4 - calculate the final cost and the smp
        # create a commitment from the results of the solver
        commitment = self.create_commitment_from_generator_data()
        # and do a dispatch on it - this is inefficient as the mip already did a dispatch. but it is cheap and avoids the hassle to implement a new manner to calculate the smp
        smp,final_cost = self.do_economic_dispatch(commitment)
        # finally, return not only the plants allocation, but also the specific generator data and the final cost
        if self.verbosity > -1:
            print ('uc solved')
        #import ipdb;ipdb.set_trace()
        return [self.all_generators_by_name.values(),final_cost,smp,self.excess_res_power]
