"""
This module handles unit commitment. It is the main driver of the wmsim application.
The heuristic unit commitment method used here is based on EPL techniques and specified in:
"Enhanced priority list unit commitment method for power systems with a high share of renewables,
E. Delarue, D. Cattrysse, W.  D'haeseleer, TME WORKING PAPER - Energy and Environment, 2013"

The algorithm comprises of the following steps:
    1. Rank power plants according to operation cost
    2. Activate the lowest number of power plants, able to meet demand and reserve requirement (account for minimum operating points)
    3. Correct for minimum up and down time violations
    4. Activate additional power plants if needed
    5. Turn off plants across an entire uptime range if possible
    6. Turn off plants in specific hours where this improves the overall solution
    7. Dispatch activated power plants and calculate costs

Beyond the initial algorithm, must_run, and partial/full plant availability constraints
have been introduced into the model
"""
import copy
import numpy
import pandas
import class_library
import multiprocessing

class UnitCommit:
    """
    This is the Unit Commit class. It is composed of all needed functions to implement the UC algorithm.
    """
    def __init__(self, reserve_margins, backup_reserves, generator_long_term_availability, uc_verbosity, use_multiprocessing):
        """
        Initialize the class with empty lists.

        The reserve_margin represents the percentage of demand fluctuation that online plants
        must be able to meet at any time

        The backup_reserve_margin is an internal algorithm variable representing the power
        to be reserved in hydro power plants (so that some hydro plants will not be allocated),
        as a percentage of the forecasted demand each hour.
        These plants can be used anyway, if the algorithm cannot find a better solution later on,
        but they are not used to find the initial plant allocation (thus kept as backup)

        The self.generators list will have the same order as the agents list, so agents will have
        the same indexes as the plants they own. This index is used extensively within wmsim.
        """
        self.use_multiprocessing = use_multiprocessing
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
        # this is the final commitment & dispatch list. it will get calculated later and will contain a plant_allocation object for each period
        self.current_commitment = "placeholder"
        # if, due to demand being very low or too many active plants during the last day,
        # the problem is unsolvable while also respecting their MUT, record the excess power here
        # and, if RES power is the reason fo such low demand, reduce that res power so that the problem is solvable
        # the reduction will be done after solving and a failure to find RES there will result to a crash
        self.excess_res_power = "placeholder"
        self.periods = "placeholder"
        self.power_cost = "placeholder"
        self.calculated_commitments_dispatch_costs = "placeholder"

        # set the correct functions to use in case of multiprocessing
        self.calculate_final_cost = self.calculate_final_cost_singleprocessing
        if self.use_multiprocessing:
            self.calculate_final_cost = self.calculate_final_cost_multiprocessing

        self.special_circumstances_log = []

    def get_available_generators(self, agents, day):
        """
        populates a list of generators available to use in the UC solution
        """
        generators = [agent.plant for agent in agents if \
                        (agent.plant.name not in self.generator_long_term_availability.columns) or (self.generator_long_term_availability.loc[day.year,agent.plant.name] > 0)]
        return generators

    def calculate_demand_margins(self):
        """
        Calculates the required demand + reserve upper & lower margins
        """
        # they need to be rounded to avoid equality errors later on
        self.day_demand.loc[:,"lower_reserves_margin"] = numpy.round(self.day_demand.loc[:,"demand"] - self.reserve_requirements.loc[:,'FCR_downwards_min'] - self.reserve_requirements.loc[:,'FRR_downwards_min'],3)
        self.day_demand.loc[:,"upper_reserves_margin"] = numpy.round(self.day_demand.loc[:,"demand"] + self.reserve_requirements.loc[:,'FCR_upwards_min'] + self.reserve_requirements.loc[:,'FRR_upwards_min'],3)

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
        # reset excess_res_power
        self.excess_res_power = pandas.Series(0,index=numpy.arange(self.periods))
        # populate the dataframe with the cost of power
        self.power_cost = self.calculate_power_cost()
        # reset the calculated_commitments_dispatch_costs series
        self.calculated_commitments_dispatch_costs = pandas.Series(name='cost',dtype='float64')
        # reset the special_circumstances_log
        self.special_circumstances_log = []

    def index_all_generators_by_name(self):
        """
        Return a dictionary with all the generator names and objects, so they can be refered to by name
        """
        generators_dict = {}
        for plant in self.all_generators:
            generators_dict.update({plant.name:plant})
        return generators_dict

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
        # round all values to 3 decimals
        power_cost.loc[:,'mw_unit_cost'] = numpy.round(power_cost.loc[:,'mw_unit_cost'].values.tolist(),3)
        # fix the df index
        power_cost.reset_index(drop=True,inplace=True)
        # and return
        return power_cost

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

    def update_commitment_using_one_generator_data(self,plant,commitment):
        """
        Updates a plant commitment as per input generators' online_data
        """
        for period in self.day_demand.index:
            if plant.online_data.loc[period,"online"] != commitment.commitment_values.loc[period,plant.name]:
                if plant.online_data.loc[period,"online"] == 1:
                    commitment.add_plant([period],plant)
                else:
                    commitment.remove_plant([period],plant)
        return commitment

    def update_plant_online_status_by_commitment(self):
        """
        Updates plant.online_data.online based on self.current_commitment
        We don't care about power dispatch at this point so generator power is set to 0
        """
        # for each plant including virtual
        for plant in self.all_generators:
            plant.online_data.loc[0:23,'online'] = self.current_commitment.commitment_values.loc[:,plant.name]
            plant.update_uptime_downtime_data()

    def get_generators_to_reserve_by_index(self):
        """
        Returns the indices of a small number of fast plants, equal to self.backup_reserves
        These will be reserved from an initial commitment, to gain flexibility for recommiting later
        Imports are never reserved
        """
        # reserve a small percentage of fast plants to gain flexibility later, imports are never reserved
        fast_plant_indices = sorted(numpy.arange(len(self.generators)), \
                        key=lambda plant_index: \
                        ((self.generators[plant_index].min_uptime + self.generators[plant_index].min_downtime),\
                        -self.generators[plant_index].cost_metric_profit, self.generators[plant_index].kind in ['imports']))

        # caveat! plants that cannot be recommitted could be reserved
        # This is unlikely thoughm and will probably never create problems even if it happens as plant reserving is nor strictly required
        fast_plant_indices = [fast_plant_indices[index] for index in range(self.backup_reserves)]
        return fast_plant_indices

    def commitment_deep_search(self, period, plants):
        """
        This function runs if add_generators_to_solve_commitment failed
        It removes the last plant from the current commitment, then checks amongst all other uncommitted plants for a solution
        """
        # we need to go deeper!
        # initially make a new allocation to test upon
        new_commitment = copy.deepcopy(self.current_commitment)

        # get the commited and noncommited plants at this point
        initial_commitment_values = self.current_commitment.commitment_values.loc[period,:]
        commited_plant_names = initial_commitment_values.loc[initial_commitment_values == 1].index.tolist()
        noncommited_plant_names = initial_commitment_values.loc[initial_commitment_values == 0].index.tolist()
        # try to remove one plant and solve again.
        # do this until we run out of plants to remove
        for last_plant_name in reversed(commited_plant_names):
            # get the name of the new plant to test removal (last committed plant)
            last_plant = self.all_generators_by_name[last_plant_name]
            # remove that plant if possible
            if last_plant.online_data.loc[period,"can_recommit"] == 1:
                new_commitment.remove_plant([period], last_plant)
                noncommited_plant_names.append(last_plant_name)
                commited_plant_names.remove(last_plant_name)

                # and iteratively commit plants until a solution is found
                for plant_name in noncommited_plant_names:
                    plant = self.all_generators_by_name[plant_name]
                    # only try to commit if possible
                    if plant.online_data.loc[period,"can_recommit"] == 1:
                        required_power = self.day_demand.loc[period,"upper_reserves_margin"] - new_commitment.commitment_power.loc[period,"pmax"]
                        min_power_margin = self.day_demand.loc[period,"lower_reserves_margin"] - new_commitment.commitment_power.loc[period,"pmin"]
                        # only commit if the plant can fit
                        if plant.available_min_power.loc[period] <= min_power_margin:
                            new_commitment.add_plant([period], plant)
                            last_name = plant.name
                            # check if a solution was found & return the commitment if so
                            if plant.available_power.loc[period] >= required_power:
                                return new_commitment
                # if no solution was found, reset the commitment values of all initially noncommited plants to make ready for next run
                for plant_name in noncommited_plant_names:
                    plant = self.all_generators_by_name[plant_name]
                    if new_commitment.commitment_values.loc[period,plant_name] == 1:
                        new_commitment.remove_plant([period], plant)
        return False

    def add_generators_to_solve_commitment(self, period, plants):
        """
        Iteratively commit the generators from the list to solve this period's commitment problem, also respecting demand & reserve requirements
        If a commitment solution is found return True, else return False
        """
        for plant in plants:
            # only use this plant if it can be recommitted this period
            if plant.online_data.loc[period,"can_recommit"] == 1:
                required_power = self.day_demand.loc[period,"upper_reserves_margin"] - self.current_commitment.commitment_power.loc[period,"pmax"]
                min_power_margin = self.day_demand.loc[period,"lower_reserves_margin"] - self.current_commitment.commitment_power.loc[period,"pmin"]
                # add the plant if it can fit
                if plant.available_min_power.loc[period] <= min_power_margin:
                    self.current_commitment.add_plant([period], plant)
                    # check if a solution was found & return true if so
                    if plant.available_power.loc[period] >= required_power:
                        return True
        # if no solution was found yet, try a deeper search & return true if this managed to solve it
        new_commitment = self.commitment_deep_search(period, plants)
        if new_commitment:
            self.current_commitment = new_commitment
            return True
            # If a solution was still not found return false
        return False

    def do_initial_commit(self):
        """
        For every period find the commitment that can satisfy demand + reserves.
        At this point we don't care about constraints other than reserves & demand.

        Also, be sure to identify the plants with the lower mut & mdt and not allocate all of them
        Instead, keep some as backup, to be used to cover possible constraints from slower plants
        As fast plants as almost always more expensive, this should not affect prices
        For Greece, such plants are hydro ones that can be cheap if water is available - but those should also not all be used simultaneously, to conserve energy for later in the year
        """
        # create a new commitment taking into account any plants already online (must run / mut restrictions)
        # virtual plants shall not be committed initially
        self.current_commitment = self.create_commitment_from_generator_data()
        # make sure the required demand & reserves can be met. If not, the only explanation is very high RES
        # In that case there should be curtailment or redispatching
        min_reserves_restriction = self.current_commitment.commitment_power.pmin - self.day_demand.lower_reserves_margin + 0.00001 # also add a small value to avoid equality issues
        if (min_reserves_restriction > 0).any():
            # this would mean that we cannot satisfy correct operation as generated power minimum  exceeds reserves lower margin!
            # keep the generators online and increase demand instead as needed, also recording the increase
            # this is done to allow energy market to reduce res production so that the problem to be solvable
            # the reduction is done post-uc-solution
            print ("WARNING! Required generated power exceeds reserves lower margin")
            print('now fixing demand and will attempt to reduce RES power after UC solve to compensate')
            # Find how much energy must be curtailed. This shall equal the positive values of min_reserves_restriction
            min_reserves_restriction.loc[min_reserves_restriction < 0] = 0
            # increase demand & reserves limits accordingly. This needs to be rounded to avoid equality errors later on
            self.day_demand.loc[:,'demand'] = self.day_demand.loc[:,'demand'] + min_reserves_restriction
            self.day_demand.loc[:,'lower_reserves_margin'] = self.day_demand.loc[:,'lower_reserves_margin'] + min_reserves_restriction
            self.day_demand.loc[:,'upper_reserves_margin'] = self.day_demand.loc[:,'upper_reserves_margin'] + min_reserves_restriction
            # save the excess energy values for lated parse
            self.excess_res_power = min_reserves_restriction.values

        # get the plants to use. separate them as reserved, nonreserved, and virtual
        reserved_generator_indices = self.get_generators_to_reserve_by_index()
        reserved_plants = [self.generators[index] for index in reserved_generator_indices]
        nonreserved_plants = [self.generators[index] for index in numpy.arange(len(self.generators)) if index not in reserved_generator_indices]
        virtual_plants = self.virtual_generators

        # for each timeperiod try to solve the commitment problem until demand & reserve requirements are fulfilled
        for period, demand in enumerate(pandas.Series(self.day_demand.loc[:,'demand'])):
            # assume there will be not problems - if no solution is found, there will be a deeper search
            plants_to_commit = nonreserved_plants
            solution_found = self.add_generators_to_solve_commitment(period,plants_to_commit)
            # if still no solution was found, try to use reserved plants in order to solve the problem
            if not solution_found:
                # first reset commitment
                self.current_commitment = self.create_commitment_from_generator_data()
                # extend the plants list
                plants_to_commit.extend(reserved_plants)
                # and retry
                solution_found = self.add_generators_to_solve_commitment(period,plants_to_commit)
            # finally, if nothing else can solve the problem, also use virtual plants and retry
            #on the case that the lower reserves margin there no possibility to meet, this margin has to be relaxed to make the problem feasible
            #if nothing was found, there is no solution for the problem: start relaxing constraints
            if not solution_found:
                print ("Warning: no possible initial commitment could be found. Trying to relax system constraints")
                # first try to relax reserves lower limit iteratively by 1 MW until a solution is found
                initial_lower_reserves_margin = self.day_demand.loc[period,'lower_reserves_margin']
                while round(self.day_demand.loc[period,'lower_reserves_margin']) < round(self.day_demand.loc[period,'demand']) and not solution_found:
                    # increase reserve lower boundary by 1
                    self.day_demand.loc[period,'lower_reserves_margin'] += 1
                    # try to solve again
                    solution_found = self.add_generators_to_solve_commitment(period,plants_to_commit)
                    print ("Relaxed reserves lower limit at period %s from %s to %s " % (period,initial_lower_reserves_margin,self.day_demand.loc[period,'lower_reserves_margin']))
                import ipdb;ipdb.set_trace()
            # if nothing was found, there is no solution for the problem :'(
            if not solution_found:
                print ("Error, no possible initial commitment could be found. Starting debugger. Is the demand correct? Starting debugger")
                import ipdb;ipdb.set_trace()
                exit()
                return False
        # if all went well this means the problem is solvable! Return true!
        return True

    def low_utilization_plant_shutdown(self):
        """
        Attempts to shutdown plants with low utilization.
        If a power plant is online for a number of periods lower than a certain factor (FMU multiplied by the plant's minimum up time MUT), it is shut down.
        """
        for plant_index, plant in enumerate(self.all_generators):
            # get the problem periods
            total_uptime = plant.get_total_uptime_duration()
            # if the plant can be recommited & eligible, shutdown
            if (plant.online_data.loc[:,"can_recommit"] == 1).all() and plant.min_uptime*plant.factor_minimum_uptime > total_uptime:
                # shut down the plant in all periods
                plant.online_data.loc[0:23,'online'] = 0
                # apply changes to plant online data
                plant.update_uptime_downtime_data()
                # apply change to commitment
                self.update_commitment_using_one_generator_data(plant,self.current_commitment)

    def fix_single_mdt_problem_starting_up(self, plant, problem_period):
        """
        This is a function used by enforce_mdt_constraint
        It tries to fix an mdt problem by starting up the plant within the MDT violation,
        also respecting system operation limits
        """
        # now get the data [readability]
        downtime = plant.online_data.loc[problem_period,'downtime']
        enabling_possible = True
        new_startup_periods = []

        # go backwards from the problem period and try to startup the plan step by step
        for step_index in range (int(downtime)):
            current_period = problem_period - step_index
            # if the plant is recommitable in this period, and valid WRT min operating point (available_min_power since we are adding plants)
            if plant.online_data.loc[current_period,'can_recommit'] == 1 and \
                        self.day_demand.loc[current_period, 'lower_reserves_margin'] >= self.current_commitment.commitment_power.loc[current_period,"pmin"] + plant.available_min_power.loc[current_period]:
                # save the change
                new_startup_periods.append(current_period)
            else:
                # a failure means impossible to do the startups
                enabling_possible = False
                break
        # if this is valid, we can return the solution. will search for new MDT problems within main
        if enabling_possible: return new_startup_periods
        else: return False

    def fix_single_mdt_problem_shutting_down(self, plant, last_offline_period_before_problem, last_online_period_before_problem):
        """
        This is a function used by enforce_mdt_constraint
        It tries to fix an mdt problem by shutting down up the plant until MDT is respected
        The shutdown is done towards the end period as much as possible, then towards the start for the rest of the periods
        """
        mdt_delta = plant.min_downtime - plant.online_data.loc[last_offline_period_before_problem,'downtime']
        # decide the periods where the plant will be shut down
        periods_to_shutdown = [last_offline_period_before_problem] # use the last_offline_period_before_problem as a starting point
        try_towards_end = 1 # auxiliary variable to determine the direction to search
        count=0
        while len(periods_to_shutdown) < mdt_delta + 1:
            count += 1
            if count > 200:
                # This catches endless loops
                print('endless loop')
                import ipdb;ipdb.set_trace()
            # try towards end if possible
            if try_towards_end == 1 and \
            periods_to_shutdown[-1] < 23 and \
            plant.online_data.loc[periods_to_shutdown[-1]+1,'can_recommit'] == 1:
                new_period_to_shutdown = periods_to_shutdown[-1]+1
            # if it is the fist time we go towards start
            elif try_towards_end == 1 and \
            plant.online_data.loc[last_online_period_before_problem,'can_recommit'] == 1 and \
            last_online_period_before_problem > -1:
                new_period_to_shutdown = last_online_period_before_problem
                try_towards_end = 0
            # if we have to continue shutting towards start
            elif try_towards_end == 0 and \
            plant.online_data.loc[periods_to_shutdown[-1]-1,'can_recommit'] == 1 and \
            periods_to_shutdown[-1]-1 > -1:
                new_period_to_shutdown = periods_to_shutdown[-1]-1
            # if none of the above triggered, we cannot enforce the mdt constraint
            else:
                print("Error, not possible to enforce mdt constraint.. Starting debugger..")
                import ipdb;ipdb.set_trace()
            # if all went well, add the new_period_to_shutdown to the list
            periods_to_shutdown.append(new_period_to_shutdown)
        # if no problem was found, return the periods to shut down, minus the initial problem period used only as reference
        return periods_to_shutdown[1:]

    def enforce_mdt_constraint(self):
        """
        Attempts to correct mdt problems.
        time_corrections_mdt brings up a plant if MDT is not respected, but the system operating boundaries can be respected.
        If it cannot do so, the plant is shut down for more periods until the MDT is respected.
        Only the 'online' dataframe column of a plant is touched,
        then the updater will be used to complete the other columns later.
        """
        for plant in self.all_generators:
            # 1. get the initial problem periods
            downtime_problem_periods = plant.find_mdt_problems()
            # part 2. fix all mdt problems
            count = 0
            while downtime_problem_periods.sum() > 0:
                count += 1
                if count > 200:
                    # This catches endless loops
                    print('mdt - endless loop')
                    import ipdb;ipdb.set_trace()

                # get the period of the first problem
                last_offline_period_before_problem = downtime_problem_periods.loc[downtime_problem_periods == 1].index[0]
                last_online_period_before_problem = last_offline_period_before_problem - plant.online_data.loc[last_offline_period_before_problem,'downtime']

                # try to start up the plant for more periods if possible
                new_startup_periods = self.fix_single_mdt_problem_starting_up(plant, last_offline_period_before_problem)
                # if there is a solution, accept it
                if new_startup_periods:
                    # update the plant online data
                    plant.online_data.loc[new_startup_periods,'online'] = 1

                # if a startup solution impossible, shut this plant down until MDT is respected
                else:
                    periods_to_shutdown = self.fix_single_mdt_problem_shutting_down(plant, last_offline_period_before_problem, last_online_period_before_problem)
                    plant.online_data.loc[periods_to_shutdown,'online'] = 0

                # the problem was solved. Update the problems list since new problems may have appeared due to the changes applied
                plant.update_uptime_downtime_data()
                downtime_problem_periods = plant.find_mdt_problems()
            # part 3. apply changes to the current commitment once each plant has no problems
            if count > 0:
                self.update_commitment_using_one_generator_data(plant,self.current_commitment)

    def fix_single_mut_problem_shutting_down(self, plant, last_online_period_before_problem, last_offline_period_before_problem):
        """
        This is a function used by enforce_mut_constraint
        It tries to fix a mut problem by shutting the plant down for all problematic periods
        The plant MDT and system operating points are also respected
        """
        periods_to_shutdown = []
        for index_to_shutdown in range (last_online_period_before_problem-last_offline_period_before_problem):
            period = last_online_period_before_problem - index_to_shutdown
            if plant.online_data.loc[period,'can_recommit'] == 1 and period > -1:
                periods_to_shutdown.append(period)
            else:
                print('time_corrections_mut is stuck. Investigate!')
                import ipdb;ipdb.set_trace()
        return periods_to_shutdown

    def fix_single_mut_problem_starting_up(self, plant, last_online_period_before_problem, last_offline_period_before_problem):
        """
        This is a function used by enforce_mut_constraint
        It tries to fix a mut problem by bringing the plant back up as needed, preferring to have it online in following periods rather than previous if possible
        The plant MDT and system operating points are respected
        """
        count = 0
        # we need to startup only in this number of periods
        mut_delta = plant.min_uptime - plant.online_data.loc[last_online_period_before_problem,'uptime']

        # decide the periods where the plant will startup
        periods_to_startup = [last_online_period_before_problem] # use the last_online_period_before_problem as a starting point
        try_towards_end = 1 # auxiliary variable to determine the direction to search
        while len(periods_to_startup) < mut_delta + 1:
            count += 1
            if count > 200:
                # This catches endless loops
                print('endless loop')
                import ipdb;ipdb.set_trace()

            # try towards end if possible
            if try_towards_end == 1 \
            and plant.get_next_downtime_period_duration(periods_to_startup[-1]) > plant.min_downtime \
            and periods_to_startup[-1] < 23 \
            and plant.online_data.loc[periods_to_startup[-1]+1,'can_recommit'] == 1 \
            and self.day_demand.loc[periods_to_startup[-1]+1, 'lower_reserves_margin'] >= self.current_commitment.commitment_power.loc[periods_to_startup[-1]+1,"pmin"] + plant.available_min_power.loc[periods_to_startup[-1]+1]:
                new_period_to_startup = periods_to_startup[-1]+1

            # if it not possible to go towards end, but is the fist time we go towards start
            elif try_towards_end == 1 \
            and plant.online_data.loc[last_offline_period_before_problem,'can_recommit'] == 1 \
            and plant.online_data.loc[last_offline_period_before_problem,'downtime'] > plant.min_downtime \
            and last_offline_period_before_problem > -1 \
            and self.day_demand.loc[last_offline_period_before_problem, 'lower_reserves_margin'] >= self.current_commitment.commitment_power.loc[last_offline_period_before_problem,"pmin"] + plant.available_min_power.loc[last_offline_period_before_problem]:
                new_period_to_startup = last_offline_period_before_problem
                try_towards_end = 0

            # if we have to continue shutting towards start
            elif try_towards_end == 0 \
            and plant.online_data.loc[periods_to_startup[-1]-1,'can_recommit'] == 1 \
            and plant.online_data.loc[periods_to_startup[-1]-1,'downtime'] > plant.min_downtime \
            and periods_to_startup[-1]-1 > -1:
                new_period_to_startup = periods_to_startup[-1]-1

            # if point by point startup could not work (mdt conflicts?), maybe it is possible to startup for the whole period the plant is offline towards either end. If not, return zero
            else:
                # try towards end
                next_downtime_period_duration = plant.get_next_downtime_period_duration(periods_to_startup[-1])
                if next_downtime_period_duration > -1:
                    # initialize new_period_to_startup as empty list
                    new_period_to_startup = []
                    # if no new startup happened after periods_to_startup[-1], next_downtime_period_duration == inf. If so, try to start the plant for the remainder of this day
                    if next_downtime_period_duration == numpy.inf:
                        new_period_to_startup = numpy.arange(periods_to_startup[0]+1,23)
                    else:
                        next_online_period = periods_to_startup[-1] + next_downtime_period_duration + 1
                        new_period_to_startup = numpy.arange(periods_to_startup[0]+1,next_online_period)
                    # if a solution was found and the plant is recommitable during those periods and if pmin of demand is below limits for those periods
                    if len(new_period_to_startup) > 0 and (plant.online_data.loc[new_period_to_startup,'can_recommit'] == 1).all() and \
                        (self.day_demand.loc[new_period_to_startup, 'lower_reserves_margin'] >= self.current_commitment.commitment_power.loc[new_period_to_startup,"pmin"] + plant.available_min_power.loc[new_period_to_startup]).all():
                            # no need to do anything else. return the solution
                            return new_period_to_startup
                # if it did not work out, try towards front
                first_offline_period_before_problem = last_offline_period_before_problem - plant.online_data.loc[last_offline_period_before_problem,'downtime'] + 1
                if first_offline_period_before_problem > -1:
                    new_period_to_startup = numpy.arange(first_offline_period_before_problem,last_offline_period_before_problem+1)
                    if (plant.online_data.loc[new_period_to_startup,'can_recommit'] == 1).all() and \
                        (self.day_demand.loc[new_period_to_startup, 'lower_reserves_margin'] >= self.current_commitment.commitment_power.loc[new_period_to_startup,"pmin"] + plant.available_min_power.loc[new_period_to_startup]).all():
                            # no need to do anything else. return the solution
                            return new_period_to_startup
                # if none of the above triggered, we cannot enforce the mdt constraint by starting up. return an empty list
                return []
            # if all went well, add the new_period_to_shutdown to the list
            if new_period_to_startup > -1:
                periods_to_startup.append(new_period_to_startup)
        # if no problem was found, return the periods to shut down, minus the initial problem period used only as reference
        return periods_to_startup[1:]

    def enforce_mut_constraint(self):
        """
        Attempts to correct mut problems
        time_corrections_mut fixes the problem of a plant's MUT not being respected,
        by bringing the plant up for enough periods if the system operating boundaries
        can be respected and the new MDT's are okay.
        If not, the plant is shut down until MUT is respected.
        We only modify one plant at a time, and only apply the changes
        to the plants allocation when the modification is certain to be applied.
        """
        for plant in self.all_generators:
            # part 1. initially identify uptime problems
            uptime_problem_periods = plant.find_mut_problems()
            # part 2. fix all mut problems
            count = 0
            while uptime_problem_periods.sum() > 0:
                count += 1
                if count > 200:
                    # This catches endless loops
                    print('endless loop')
                    import ipdb;ipdb.set_trace()
                # get the period of the first problem
                last_online_period_before_problem = uptime_problem_periods.loc[uptime_problem_periods == 1].index[0]
                last_offline_period_before_problem = last_online_period_before_problem - plant.online_data.loc[last_online_period_before_problem,'uptime']
                # try to start up the plant for more periods if possible
                new_startup_periods = self.fix_single_mut_problem_starting_up(plant, last_online_period_before_problem, last_offline_period_before_problem)
                # if there is a solution, accept it
                if len(new_startup_periods) > 0:
                    # update the plant online data
                    plant.online_data.loc[new_startup_periods,'online'] = 1
                # if a startup solution impossible, shut this plant down until MUT is respected
                else:
                    periods_to_shutdown = self.fix_single_mut_problem_shutting_down(plant, last_online_period_before_problem, last_offline_period_before_problem)
                    plant.online_data.loc[periods_to_shutdown,'online'] = 0

                # the problem was solved. Update the problems list since new problems may have appeared due to the changes applied
                plant.update_uptime_downtime_data()
                uptime_problem_periods = plant.find_mut_problems()
            # part 3. apply changes to the current commitment once each plant has no problems
            if count > 0:
                self.update_commitment_using_one_generator_data(plant,self.current_commitment)

    def find_power_shortage_periods(self):
        """
        Finds periods where the plant allocation is insufficient and returns them as a list of tuples
        """
        shortages = pandas.Series(0,index=self.current_commitment.commitment_power.index,name='power_shortage')
        for period in shortages.index:
            power_shortage = self.day_demand.loc[period,'upper_reserves_margin'] - self.current_commitment.commitment_power.loc[period,'pmax']
            if power_shortage > 0:
                shortages.loc[period] = power_shortage
        return shortages

    def check_if_a_plant_can_be_started_considering_all_constraints(self, period, plant):
        """
        Checks if a plant can start at the given period, respecting all system & plant constraints
        """
        # initialize with a denial
        startup_possible = 0

        # if the plant can be recommited and the system min reserve respected, startup the plant so long as not mdt problems crop up
        if plant.online_data.loc[period,'online'] == 0 and plant.online_data.loc[period,'can_recommit'] == 1 and \
                        self.day_demand.loc[period, 'lower_reserves_margin'] >= self.current_commitment.commitment_power.loc[period,"pmin"] + plant.available_min_power.loc[period]:
            # provisionally apply the change
            plant.online_data.loc[period,'online'] == 1
            # test for mdt problems
            mdt_problems = plant.find_mdt_problems()
            # revert the change - others shall apply it if required
            plant.online_data.loc[period,'online'] == 0
            if mdt_problems.sum() == 0:
                # change the response to true as the plant may start
                startup_possible = 1
        # and return the verdict
        return startup_possible

    def start_up_more_required_plants(self):
        """
        Attempt to fix power shortage problems.
        If there are power shortages, this function starts more plants to fix them.
        """
        # get the problematic periods
        shortages = self.find_power_shortage_periods()
        # and fix them by starting more plants one by one in each period
        for period in shortages.loc[shortages>0].index:
            required_power = shortages.loc[period]
            # try with all plants bar virtual ones
            for plant in self.generators:
                # check if the plant can be started
                startup_possible = self.check_if_a_plant_can_be_started_considering_all_constraints(period, plant)
                if startup_possible:
                    plant.online_data.loc[period,'online'] == 1
                    self.current_commitment.add_plant([period],plant)
                    # if a new plant started up, update the required power
                    required_power -= plant.available_power.loc[period]
                    # check if a change was sufficient to cover the shortage to stop
                    if required_power <= 0:
                        break
            # check if a change was sufficient to cover the shortage to stop. if not, also use virtual plants
            if required_power > 0:
                for plant in self.virtual_generators:
                    # check if the plant can be started
                    startup_possible = self.check_if_a_plant_can_be_started_considering_all_constraints(period, plant)
                    if startup_possible:
                        print ('Warning! Adding a virtual plant in start_up_more_required_plants')
                        plant.online_data.loc[period,'online'] == 1
                        self.current_commitment.add_plant([period],plant)
                        # if a new plant started up, update the required power
                        required_power -= plant.plant.available_power.loc[period]
                        # check if a change was sufficient to cover the shortage to stop
                        if required_power <= 0:
                            break
                # check for unsovable problems due to lack of capacity
                if required_power > 0:
                    print ('Error. Not enough capacity to meet demand. Starting debugger')
                    import ipdb;ipdb.set_trace()

    def shutdown_excess_plants(self):
        """
        Attempt to fix excess power inefficiencies by shutting down plants if possible.
        """
        # b1 - shutting down over entire time window
        self.shut_down_plants_over_entire_time_window()
        # b2 - shutting down during specific periods
        self.shut_down_plants_during_specific_periods()

    def shut_down_plants_over_entire_time_window(self):
        """
        Shut down a plant within a whole working period if it is not needed in that period.
        Checks all plants and all periods each plant is online.
        """
        # try to remove plants starting from the last ones - virtual and most expensive first
        for plant in reversed(self.all_generators):
            # to try to remove, the plant needs to be online at least for one period and recommittable when online
            plant_online_periods = plant.online_data.loc[plant.online_data.loc[:,'online']==1].loc[0:23,:].index
            if len(plant_online_periods) > 0 and plant.online_data.loc[plant_online_periods,'can_recommit'].all():
                # Also system constraints may not be violated - assume the plant is shutdown and check if this is true
                self.current_commitment.remove_plant(plant_online_periods,plant)
                if (self.day_demand.lower_reserves_margin - self.current_commitment.commitment_power.pmin > 0).all() and \
                                (self.current_commitment.commitment_power.pmax - self.day_demand.upper_reserves_margin > 0).all():
                    # if the system reserve constraints are not violated accept the change, else revert it
                    plant.online_data.loc[plant_online_periods,'online'] = 0
                    plant.update_uptime_downtime_data()
                else:
                    self.current_commitment.add_plant(plant_online_periods,plant)

    def shut_down_plants_during_specific_periods(self):
        """
        Try to shutdown a plant at specific hours if it proves more economical, and provided the other plants are enough to power the system
        Calculate costs by making a final dispatch
        Runs twice on every period for every commited plant on these periods, first parsing periods descending, then ascending.
        The 'reverse searching' trick helps to shutdown plants as much as possible
        """
        # create the index list before parsing period by period - this is a collation of the period indices in reverse and normally
        index_list = self.day_demand.index[::-1].append(self.day_demand.index[1:])

        # create a flag with the direction, in order to correctly check for system constraint violations
        direction = 'backwards'

        for period in index_list:
            # detect direction changes. this happens exactly at period 0
            if period == 0: direction = 'forwards'

            # get the online plants at this period
            online_plants_names = self.current_commitment.commitment_values.loc[period].loc[self.current_commitment.commitment_values.loc[period] == 1].index
            online_plants = [self.all_generators_by_name[plant_name] for plant_name in online_plants_names]

            # and try to shut them down starting from the most expensive (in reverse) if possible
            for plant in reversed(online_plants):
                # only try if the plant is recomittable
                # if checking backwards, the uptime needs to be larger than min_uptime
                # if checking forwards, the uptime again needs to be larger than min_uptime, but the uptime needs to be found first
                # in both cases, mut/mdt will not be violated if we shut the plant down for the period.
                # in addition, we need to check if the min commitment_power of the period violates the system pmin constraint
                if plant.online_data.loc[period,"can_recommit"] == 1 and \
                    ((direction == 'backwards' and plant.online_data.loc[period+1,'online'] == 0 and plant.online_data.loc[period,'uptime'] > plant.min_uptime) or \
                    (direction == 'forwards' and plant.online_data.loc[period-1,'online'] == 0 and plant.get_next_uptime_period_duration(period) > plant.min_uptime)) and \
                    (self.current_commitment.commitment_power.loc[period,'pmax'] - plant.available_power.loc[period] >= self.day_demand.loc[period,'upper_reserves_margin']):
                        self.shutdown_plant_if_cheaper(plant,period)

    def shutdown_plant_if_cheaper(self,plant,period):
        """
        Checks the commitment final costs with plant online & offline, and shuts it down if cheaper
        """
        old_cost = self.get_commitment_cost(self.current_commitment)
        # apply the new change and get new cost
        plant.online_data.loc[period,'online'] = 0
        self.current_commitment.remove_plant([period],plant)
        new_cost = self.get_commitment_cost(self.current_commitment)
        # compare costs
        if old_cost > new_cost:
            # if the new solution is better, keep the change and recalculate the plant uptime/downtime
            plant.update_uptime_downtime_data()
        else:
            # if it's not that good, revert the change
            plant.online_data.loc[period,'online'] = 1
            self.current_commitment.add_plant([period],plant)

    def get_commitment_cost(self, commitment):
        """
        Gets the final cost of a commitment
        Checks if this was calculated before, and if not, runs an economic dispatch to get it
        """
        # if the cost was calculated before, get it
        if commitment.commitment_hash() in self.calculated_commitments_dispatch_costs:
            return self.calculated_commitments_dispatch_costs.loc[commitment.commitment_hash()]
        # if not, calculate it
        else:
            return self.do_economic_dispatch(commitment=commitment)[1]


    def do_economic_dispatch_binary(self, commitment):
        """
        Unutilized because slower than the other version
        """

        smp = pandas.Series(index=self.day_demand.index,name='smp')

        for period in self.day_demand.index:
            # 1. get some boundaries for smp. Those can be the min & max costs of the cost list
            # filter out all noncommited plants as well as the virtual plant from the power cost list
            commited_plants = commitment.commitment_values.loc[period,:].loc[commitment.commitment_values.loc[period,:] > 0].index.tolist()
            power_cost = self.power_cost.loc[self.power_cost.loc[:,'generator'].isin(commited_plants)]
            real_power_costs = power_cost.loc[power_cost.loc[:,'mw_unit_cost'] < 1E20].loc[:,'mw_unit_cost']
            # the real_power_costs list is sorted so we know the upper/lower boundary values
            smp_max = real_power_costs.iloc[-1]
            smp_min = real_power_costs.iloc[0]

            # 2. start binary search loop
            # initialize smp
            smp.loc[period] = numpy.round((smp_max+smp_min)/2,3)
            # check smp for feasibility
            for plantname in commited_plants:
                # filter out all segments under this smp
                active_power_costs = power_cost.loc[power_cost.loc[:,'mw_unit_cost'] <= smp.loc[period]]
                # and filter out all segments that have a higher segment of the same generator
                # sort segments by generator
                active_power_costs = active_power_costs.sort_values(['generator','segment_pmax'],ascending=False)
                active_power_costs = active_power_costs.drop_duplicates(subset='generator', keep='first')
                # get the available power boundary defined by the last (most expensive) unit segment
                max_power = active_power_costs.loc[:,'segment_pmax'].sum()
                min_power = active_power_costs.iloc[:-1].loc[:,'segment_pmax'].sum() + active_power_costs.iloc[-1].loc['segment_pmin']

                # check if a solution was found
                # a solution is valid only if the last bid can be partially or fully cleared
                if max_power >= self.day_demand.loc[period,'demand'] and min_power <= self.day_demand.loc[period,'demand']:
                    # if we found the solution, apply the power to all generators
                    for index in active_power_costs.index[:-1]:
                        plant = self.all_generators_by_name[active_power_costs.loc[index,'generator']]
                        plant.online_data.loc[period,'power'] = active_power_costs.loc[index,'segment_pmax']
                    # the last generator may have less generation than segement_pmax
                    plant = self.all_generators_by_name[active_power_costs.loc[active_power_costs.index[-1],'generator']]
                    plant.online_data.loc[period,'power'] = active_power_costs.loc[index,'segment_pmax'] - (max_power - self.day_demand.loc[period,'demand'])
                    # stop here
                    break
                # if available power does not suffice to meed demand, increase the smp
                elif max_power < self.day_demand.loc[period,'demand']:
                    smp.loc[period] = numpy.round((smp_max+smp.loc[period])/2,3)
                # if available power is more than the demand, decrease the smp
                elif min_power > self.day_demand.loc[period,'demand']:
                    smp.loc[period] = numpy.round((smp.loc[period]+smp_min)/2,3)

        # now that the dispatch was completed, also get the real cost of the system
        # this cost will not equal smp * demand because (1) startup costs were not taken into account and (2) the problem is non-convex
        # at such an instance, all commited generators with energy unit cost higher than the smp will be remunerated at their bid
        final_cost = self.calculate_final_cost(smp)

        # save the final cost to the related series, to cut down on calculations
        self.calculated_commitments_dispatch_costs.loc[self.current_commitment.commitment_hash()] = final_cost

        # and return the data
        return smp,final_cost

    def do_economic_dispatch(self, commitment):
        """
        Do an economic dispatch on the commitment given, apply it to the generators and find the marginal price (price to get 1 more unit of power)

        This is the most expensive function of the module (and one of the two worst in wmsim).
        It should be possible to optimize its performance by unpacking the plant objects into arrays and then writing the main loop in C
        """
        # set generation of all online plants to available_min_power and for each period and online plant get all power/cost segments
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
                # test
                # if (plant.online_data.loc[period,"online"] == 1 and plant.online_data.loc[period,'power'] < plant.available_min_power.loc[period]) or plant.online_data.loc[period,'power'] < 0:
                    # print ('A unit was scheduled below pmin or 0. starting debugger')
                    # import ipdb;ipdb.set_trace()

        # now that the dispatch was completed, also get the real cost of the system
        # this cost will not equal smp * demand because (1) startup costs were not taken into account and (2) the problem is non-convex
        # at such an instance, all commited generators with energy unit cost higher than the smp will be remunerated at their bid
        final_cost = self.calculate_final_cost(smp)

        # save the final cost to the related series, to cut down on calculations
        self.calculated_commitments_dispatch_costs.loc[self.current_commitment.commitment_hash()] = final_cost
        # and return the data
        return smp,final_cost

    def calculate_final_cost_singleprocessing(self,smp):
        """
        Calculates the final cost of the dispatch using a single core/thread
        """
        final_cost = 0
        for plant in self.all_generators:
            final_cost += plant.calculate_one_day_income(smp)
        return final_cost

    def calculate_final_cost_multiprocessing(self,smp):
        """
        Calculates the final cost of the dispatch using as many cores/threads as specified in parallel_jobs
        Too many jobs may slow calculations as the overhead becomes greater than the benefit.
        """
        # get the number of parallel jobs. This is capped to 3 max jobs as testing shows that more do not yield performance improvements.
        parallel_jobs = min(3,int(multiprocessing.cpu_count()))
        # get the no of all generators
        batch_interval = len(self.all_generators)
        # split the plant objects in parralel_jobs number of parts
        plants_split = [self.all_generators[job_index*batch_interval // parallel_jobs: (job_index+1)*batch_interval // parallel_jobs] for job_index in range(parallel_jobs)]
        # set final cost to 0
        final_cost = 0

        # start calculating final costs in a multiproc manner
        # create a queue for the results
        multiprocessing_costs = multiprocessing.SimpleQueue()
        multiprocessing_jobs = []
        for plants_batch in plants_split:
            new_job = multiprocessing.Process(target=self.calculate_plants_one_day_income_sum_multiprocessing, args=(plants_batch,smp,multiprocessing_costs))
            multiprocessing_jobs.append(new_job)
            new_job.start()
        #  get the costs
        for job in multiprocessing_jobs:
            final_cost += multiprocessing_costs.get()
        # and finalize the jobs
        for job in multiprocessing_jobs:
            job.join()
        return final_cost

    def calculate_plants_one_day_income_sum_multiprocessing(self,plants,smp,multiprocessing_costs):
        """
        Calculates the sum of the daily income for the plants and smp given.
        The result is being put within the multiprocessing_costs Queue
        """
        cost = 0
        for plant in plants:
            cost += plant.calculate_one_day_income(smp)
        multiprocessing_costs.put(cost)

    def test_solution(self):
        """
        Check that generation equals demand as needed
        """
        generation = pandas.Series(0,index=self.day_demand.loc[0:23,"demand"].index)
        for agent in self.single_plant_agents:
            generation += agent.plant.online_data.loc[0:23,'power']
        if not numpy.isclose(generation.sum(),self.day_demand.loc[:,"demand"].sum()):
            print("Generation does not match demand. Starting debugger")
            import ipdb;ipdb.set_trace()

    def test_solution_generators(self):
        """
        Check that generation is positive or zero for all generators
        """
        for index,agent in enumerate(self.single_plant_agents):
            if (agent.plant.online_data.loc[0:23,'power'] < 0).any():
                print("Generation is negative for unit "+agent.plant.name+", index "+index+". Starting debugger")
                return -1
            return 1

    def calculate_unit_commit(self,day):
        """
        This is the main function of the class. Does what it says on the tin.
        """
        if self.verbosity > -1:
            print ('Starting new unit commit calculation')
        # step 0 - update the uc data to get ready for running the algorithm
        self.update_uc_data(day)
        # Step 1 - Calculate the initial allocation of plants
        if self.verbosity > 0:
            print('Finding initial allocation')
        self.do_initial_commit()
        # apply the allocation to all plants
        self.update_plant_online_status_by_commitment()
        # Step 2 - Apply min uptime and min downtime corrections
        if self.verbosity > 0:
            print('Applying fmu,mut,mdt corrections')
        # A - use FMU to shut down very low use plants (no sys limits taken into account)
        self.low_utilization_plant_shutdown()
        # B - fix the problem of a plants MDT not being respected (taking into account also sys reserves & plant touchability)
        self.enforce_mdt_constraint()
        # C - fix the problem of a plants MUT not being respected (taking into account also sys reserves, plant touchability & mdts)
        self.enforce_mut_constraint()
        # step 3 - Activation and shutting down of additional power plants if necessary
        # a - starting up
        if self.verbosity > 0:
            print ('Starting more plants if needed')
        self.start_up_more_required_plants()
        # b1 - shutting down over entire range
        if self.verbosity > 0:
            print ('Shuting down excess plants')
        self.shutdown_excess_plants()
        # step 4 get the final dispatch based on the allocations provided
        # on the event that there is no final dispatch abort the calculation
        if self.verbosity > 0:
            print ('Finding final dispatch and costs')
        # calculate all costs (finding the final dispatch is implied)
        smp, final_cost = self.do_economic_dispatch(commitment=self.current_commitment)
        if type(final_cost) == str:
            return final_cost
        # test solution
        test = self.test_solution_generators()
        if test < 0:
            import ipdb;ipdb.set_trace()

        # finally, return not only the plants allocation, but also the specific generator data and the final cost
        if self.verbosity > -1:
            print ('UC solved')
        return [self.all_generators, final_cost, smp, self.excess_res_power]
