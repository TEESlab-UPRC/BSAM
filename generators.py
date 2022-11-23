"""
This module containes the generator class and subsequent functions in order to model a generator.
A generator is a discrete electricity-generating factory that can be of one of many types,
and have its own attributes and specifications.
Generators are the main drivers of electricity production in the model and their data is
accessed and modified very frequently.
"""
import numpy
import pandas
import class_library


class Generator:
    """
    Represents a generator
    It contains not only a generator's initial data, but also dispatch data (self.online_data)
    and cost metrics (being the cost per unit to run the generator at the median of its possible utilization - (Pmax-Pmin)/2 )

    Cost metrics also take into account a profit factor specified by the agent, meaning
    a percentage increase in running costs, with the difference between
    the "base" costs and the "actual" costs becoming the plant agents profit.

    Also the plant use costs are linearized around 8 points, thus separating the use
    area into a number of linear-cost segments. To do this we choose equal segments in the space
    between Pmin to Pmax and calculate the quadratic cost at their medians.

    Finally many generator-specific functions are implemented within this class, such as
    data updaters and mut/mdt conflict finders.
    """
    def __init__(self, generator_data, linearized_segments_number, market_data):
        # this is the market_data dataframe, containing various market rules and limitations
        self.market_data = market_data
        # constant attributes
        self.name = str(generator_data.Name)
        self.kind = str(generator_data.kind)
        self.basin = str(generator_data.basin)
        self.pmax = round(float(generator_data.Pmax),3)
        self.pmin = round(float(generator_data.Pmin),3)
        self.start_costs = [generator_data.cc, generator_data.hc]  # [cc, hc] hot & cold start costs
        self.tcold = generator_data.tcold # time to transit to cold state
        self.ramp_up = generator_data.ramp_up
        self.ramp_down = generator_data.ramp_down
        self.min_uptime = generator_data.MUT
        self.min_downtime = generator_data.MDT
        self.factor_minimum_uptime = generator_data.factor_minimum_uptime
        self.must_run = generator_data.must_run
        self.cost_funct_params = [generator_data.a, generator_data.b, generator_data.c]  # [a,b,c] cost coefficients
        # apply price shifting for specific techs
        if self.kind in ['lignite-st']:
            self.cost_funct_params = (numpy.array(self.cost_funct_params)*(1 + market_data.loc['lignite_price_shift_percent','data1'])).tolist()
        elif self.kind in ['nat-gas-ccgt','nat-gas-ocgt','nat-gas-st']:
            self.cost_funct_params = (numpy.array(self.cost_funct_params)*(1 + market_data.loc['nat_gas_price_shift_percent','data1'])).tolist()
        self.heat_rates = [generator_data.a_hr, generator_data.b_hr, generator_data.c_hr]  # [a,b,c] heat rate coefficients
        self.carbon_coefficient = generator_data.carbon_coefficient # (44/12*Carbon_content)/Gross_Calorific_Value_KJ
        self.carbon_price = 0 # E/Tonne - initialize at 0

        # the initial state of the plant. this represents uptime if positive & downtime if negative, while 0 means it has just shutdown
        self.inist = generator_data.inist
        # the number of segments around which we will linearize the plant mc costs
        self.linearized_segments_number = linearized_segments_number
        # mutable attributes
        # this assumes 24 periods (1h each)

        # uptime & downtime columns are cumulative - the index is (-1 to 24 = 26 total) hours to solve out-of-bounds problems
        self.online_data = pandas.DataFrame(0, index = pandas.Index(numpy.arange(-1,25), name = 'timeperiod'), \
                                                        columns = ['online','power','uptime','downtime','can_recommit'])
        # units start recommitable
        self.online_data.loc[:,'can_recommit'] = 1
        # convert the power column to float
        self.online_data['power'] = self.online_data.power.astype(float)
        # available_power is a pandas series containing the pmax of the plant on any given period based on plant availability. assumed max initially
        self.available_power = pandas.Series(self.pmax, index = pandas.Index(numpy.arange(-1,25)), name = 'period')
        # available pmin must also be scaled
        self.available_min_power = pandas.Series(self.pmin, index = pandas.Index(numpy.arange(-1,25)), name = 'period')
        # since a unit that is partially available may need to use one value for pmin/pmax for a period (eg to calculate bid segments), get and update those values
        self.available_pmax = self.pmax
        self.available_pmin = self.pmin
        self.update_available_pmax_pmin()

        # at creation, update the online data table with initial uptime & downtime
        if self.inist < 0: self.online_data.loc[-1,'downtime'] = -self.inist
        else: self.online_data.loc[-1,'uptime'] = self.inist
        # these final attributes are result of agent interaction concerning the
        # plant profit and availability - those will change every day
        # we define as profit_factor the factor with which we multiply the 'b' variable of the operating cost polyonym (ie the linearized MC)
        # as a result of the agent action. a larger value here implies large profits for the agent (1 = 100% = no profit)
        self.profit_factor = 1
        # we define as fuel factor the factor with which we multiply 'b' variable of the operating cost polyonym
        # as a result of the fluctuation in fuel prices. this is relevant only for nat. gas plants, as lignite prices remain the same
        # hydro prices are calculated separately market-wide & pv,wind prices are also applied market-wide
        self.fuel_factor = 1
        # effective availability is the product of agent interaction for the plants availability.
        # it will get set when available_power is set in the market_data module
        # self.effective_availability = None  #don't use this yet
        # cost metrics
        self.cost_metric = 0  # to be updated after availability has been taken into account
        self.cost_metric_profit = 0  # to be updated after profit factor has been taken into account
        self.marginal_costs = self.calculate_mc_segments('cost')  # list of tuples (pmin,pmax,cost)
        self.marginal_costs_by_profit_daily = ()  # this is filled each day with the marginal_costs + profit factor
        self.special_extra_costs = [0]*24  # this list represents extra costs for the day, such as costs of violating MUT, one for each period
        # this is for hydro plants. it represents a difference between the expected produced energy for this month vs the actual one
        self.produced_energy_difference = 0
        # save the online data for use later on - This an empty df so that we can concat with it easily
        self.saved_online_data = pandas.DataFrame()
        # this variable is used to signify that the plant has closed down, and records the date this happened
        self.close_down = [False,False]

    def __str__(self):
        return "plant name: %s" % (self.name)

    def update_available_pmax_pmin(self):
        """
        Updates the available pmax and pmin of the unit, assuming that if the unit is partially available
        pmax is the largest possible, while pmin is the lowest possible
        """
        self.available_pmax = self.available_power.max()
        self.available_pmin = self.available_min_power.min()

    def calculate_cost_metric(self, profit_type, powerlevel_percent=50,mode='running'):
        """
        This is used for ranking plants. The metric equals running costs at a specified powerlevel.
        Default is powerlevel at 50%
        If the profit_type equals 'income', also use the profit_factor specified to modify the cost.
        If the mode is variable, only take the b coefficient into acount rather than a,b,c
        """
        # decide on a profit factor
        if profit_type == 'income': profit_factor = self.profit_factor
        elif profit_type == 'cost': profit_factor = 1
        else:
            print("there is no such profit type. use 'income' or 'cost'")
            exit()
        power_lvl = self.available_pmin + (self.available_pmax - self.available_pmin)*powerlevel_percent/100.0
        if mode == 'variable':
            cost = self.variable_cost_per_MW(power_lvl, profit_factor)
        elif mode == 'running':
            cost = self.running_cost_per_MW(power_lvl, profit_factor)
        return cost

    def calculate_carbon_cost(self,power_lvl,return_value='total_carbon_cost'):
        """
        Calculates the total CO2 cost for the plant at the given power level
        """
        total_carbon_cost, total_emissions, carbon_emissions_per_mw, carbon_cost_per_mw = 0, 0, 0, 0
        if power_lvl > 0 and self.carbon_coefficient > 0 and self.carbon_price:
            carbon_emissions_per_mw = (self.heat_rates[0]/power_lvl + self.heat_rates[1] + self.heat_rates[2]*power_lvl)*self.carbon_coefficient
            carbon_cost_per_mw = carbon_emissions_per_mw * self.carbon_price
            total_emissions = carbon_emissions_per_mw * power_lvl
            total_carbon_cost = carbon_cost_per_mw * power_lvl
        if return_value == 'total_carbon_cost': return_value = total_carbon_cost
        elif return_value == 'total_carbon_emissions': return_value = total_emissions
        elif return_value == 'carbon_cost_per_mw': return_value = carbon_cost_per_mw
        elif return_value == 'carbon_emissions_per_mw': return_value = carbon_emissions_per_mw
        return return_value

    def running_cost_per_MW(self, power_lvl, profit_factor):
        """
        Calculates and returns the plants running cost/MW at a specified power_level and profit_factor
        Assumes quadratic running_costs function with coefficients being those in self.cost_funct_params
        """
        # fix possible nan/inf problems when power_lvl = 0
        cost = 0
        if power_lvl > 0:
            energy_cost = profit_factor*(self.cost_funct_params[0] + self.cost_funct_params[1]*power_lvl*self.fuel_factor + self.cost_funct_params[2]*power_lvl**2)/power_lvl
            carbon_cost = self.calculate_carbon_cost(power_lvl)/power_lvl
            cost = energy_cost + carbon_cost
        # cost cannot be over the market set max. the plant is forced to not go above this
        return min(cost,self.market_data.loc['max_electricity_price','data1'])

    def variable_cost_per_MW(self, power_lvl, profit_factor):
        """
        Calculates and returns the plants variable cost/MW at a specified power_level
        Takes into account only the b coefficient of the self.cost_funct_params
        """
        # fix possible nan/inf problems when power_lvl = 0
        cost = 0
        if power_lvl > 0:
            energy_cost = profit_factor*(self.cost_funct_params[1]*power_lvl*self.fuel_factor)/power_lvl
            carbon_cost = self.calculate_carbon_cost(power_lvl)/power_lvl
            cost = energy_cost + carbon_cost
        return min(cost,self.market_data.loc['max_electricity_price','data1'])

    def calculate_mc_segments(self,profit_type):
        """
        This function linearizes the marginal costs/MW in segments_number number of segments,
        each taking up 1/segments_number of the Pmax-Pmin distance.
        It also takes into account the profit factor if required and
        returns the segments as a list of tuples (pmin,pmax,cost)
        It rounds power of each segment into the nearest integer
        """
        if profit_type == 'income': profit_factor = self.profit_factor
        elif profit_type == 'cost': profit_factor = 1
        else:
            print("there is no such profit type. use 'income' or 'cost'")
            exit()
        mc_segments = pandas.DataFrame(index=numpy.arange(int(self.linearized_segments_number)),columns=['mw_unit_cost','segment_pmin','segment_pmax','generator'])

        segment_range = round((self.pmax-self.pmin)/self.linearized_segments_number,3)
        for segment in range (int(self.linearized_segments_number)-1):
            segment_pmin = self.pmin + segment_range*segment
            segment_pmax = segment_pmin + segment_range
            # we need the mc at the median of the segment
            linearized_mc = self.running_cost_per_MW((segment_pmin+segment_pmax)/2, profit_factor)
            # if the plant is virtual, this cost is very very large
            if self.kind == 'virtual': linearized_mc = 1E20
            mc_segments.loc[segment,:] = [linearized_mc,segment_pmin,segment_pmax,self.name]
        # since the segment_range is rounded, the last segment will need to be handled separately
        # to take into account rounding errors. Thus its pmax must equal plant pmax
        segment_pmin = self.pmin + segment_range*(segment+1)
        segment_pmax = self.pmax
        linearized_mc = self.running_cost_per_MW((segment_pmin+segment_pmax)/2, profit_factor)
        if self.kind == 'virtual': linearized_mc = 1E20
        mc_segments.loc[segment+1,:] = [linearized_mc,segment_pmin,segment_pmax,self.name]

        # return, no need to sort as the per unit cost is an descending function
        return mc_segments

    def update_uptime_downtime_data(self):
        """
        Updates the uptime & downtime values of the online_data dataframe
        by using only the 'online' column as reference.
        """
        uptime_in_all_periods = []
        downtime_in_all_periods = []
        uptime = self.online_data.loc[-1,'uptime']
        downtime = self.online_data.loc[-1,'downtime']
        for index, online_status in enumerate(self.online_data.loc[0:23,'online']):
            if online_status:
                uptime += 1
                downtime = 0
            else:
                uptime = 0
                downtime += 1
            # save the data for the period
            uptime_in_all_periods.append(uptime)
            downtime_in_all_periods.append(downtime)
        # update all data in one go
        self.online_data.loc[0:23,['uptime','downtime']] = numpy.array([uptime_in_all_periods, downtime_in_all_periods]).T

    def get_next_downtime_period_duration(self, period):
        """
        Returns the duration of the downtime following the given period and until next startup
        """
        status_changes = (self.online_data.loc[0:23,'online'] - self.online_data.loc[-1:23,'online'].shift(1)).loc[0:23]
        next_startups = status_changes.loc[period:23].loc[status_changes == 1]

        # if a startup exists, calculate the duration, else return inf
        duration = numpy.inf
        if len(next_startups.index) > 0:
            if next_startups.index[0] == period:
                if len(next_startups.index) > 1:
                    duration = next_startups.index[1] - (period + 1)
            else:
                duration = next_startups.index[0] - (period + 1)
        return duration

    def get_next_uptime_period_duration(self, period):
        """
        Returns the duration of the uptime following the given period and until next shutdown
        """
        status_changes = (self.online_data.loc[0:23,'online'] - self.online_data.loc[-1:23,'online'].shift(1)).loc[0:23]
        next_shutdowns = status_changes.loc[period:23].loc[status_changes == -1]

        # if a startup exists, calculate the duration, else return inf
        duration = numpy.inf
        if len(next_shutdowns.index) > 0:
            if next_shutdowns.index[0] == period:
                if len(next_shutdowns.index) > 1:
                    duration = next_shutdowns.index[1] - (period + 1)
            else:
                duration = next_shutdowns.index[0] - (period + 1)
        return duration

    def get_total_uptime_duration(self):
        """Returns the total uptime of the plant for the day"""
        return self.online_data.loc[0:23,'online'].sum()

    def enforce_mut_mdt_restrictions_from_last_day_operation(self):
        """
        Updates the plant's online data depending on the initial state, so as not to allow mdt or mut problems
        Also updates the can_recommit column of online_data
        """
        last_day_downtime = self.online_data.loc[-1,'downtime']
        last_day_uptime = self.online_data.loc[-1,'uptime']
        change_to = 0
        last_period_to_change = -1

        # check for possible mdt/mut problems
        # shutdown required
        if last_day_downtime > 0 and last_day_downtime < self.min_downtime:
            last_period_to_change = int(self.min_downtime - last_day_downtime)
            change_to = 0
        # startup required
        elif last_day_uptime > 0 and last_day_uptime < self.min_uptime:
            last_period_to_change = int(self.min_uptime - last_day_uptime)
            change_to = 1

        # if needed, mark it as forced offline / online
        if last_period_to_change > -1:
            self.online_data.loc[0:last_period_to_change,'online'] = change_to
            self.online_data.loc[0:last_period_to_change,'can_recommit'] = 0

    def enforce_must_run_restriction(self):
        """
        Updates the plant's online data depending on its must_run status
        Also updates the can_recommit column of online_data
        """
        if self.must_run > 0:
            self.online_data.loc[0:23,'online'] = 1
            self.online_data.loc[0:23,'can_recommit'] = 0

    def find_mut_problems(self):
        """
        Returns a dataframe with all the mut violations found.
        The last period of each problem is marked with 1
        """
        # create a new series to store problems (initialize with 0)
        uptime_problem_periods = pandas.Series(0,index=self.online_data.index,name='uptime_problem_periods')
        # the last period need not be checked individually as the 25th hour is full of 0s - check everything else
        for period, uptime in self.online_data.loc[-1:23,'uptime'].items():
            if uptime > 0 and self.online_data.loc[period+1,'uptime'] == 0 and uptime < self.min_uptime and period+1 !=24:
                uptime_problem_periods.loc[period] = 1
        return uptime_problem_periods

    def find_mdt_problems(self):
        """
        Returns a dataframe with all the mdt violations found.
        The last period of each problem is marked with 1
        """
        # create a new series to store problems (initialize with 0)
        downtime_problem_periods = pandas.Series(0,index=self.online_data.index,name='downtime_problem_periods')
        # the last period need not be checked individually as the 25th hour is full of 0s - check everything else
        for period, downtime in self.online_data.loc[-1:23,'downtime'].items():
            if downtime > 0 and self.online_data.loc[period+1,'downtime'] == 0 and downtime < self.min_downtime and period+1 !=24:
                downtime_problem_periods.loc[period] = 1
        return downtime_problem_periods

    def calculate_one_day_cost(self):
        """
        Calculates the cost to run this plant for the day, considering the online data of the plant at the state it is now
        Uses the linearized mc costs of each bid.
        Also factors in startup costs
        """
        # first calculate startup cost
        startup_costs = self.calculate_startup_cost()
        # also calculate the energy cost
        # for the real cost, profit factor will equal 1 & cost will be the real cost
        energy_costs = 0
        for generated_power in self.online_data.loc[0:23].loc[self.online_data.loc[0:23,'power']>=self.available_min_power.loc[0:23],'power']:
            for segment_index in reversed(self.marginal_costs.index):
                if generated_power >= self.marginal_costs.loc[segment_index,'segment_pmin'] and \
                        generated_power <= self.marginal_costs.loc[segment_index,'segment_pmax']:
                    energy_costs += generated_power * self.marginal_costs.loc[segment_index,'mw_unit_cost']
                    break
        return energy_costs + startup_costs

    def calculate_one_day_variable_cost(self,online_data):
        """
        Calculates the hourly variable cost of this plant for the day, considering the online data of the plant at the state it is now
        Uses the linearized mc costs of each bid.
        """
        # for the real cost, profit factor will equal 1 & cost will be the real cost
        variable_costs = pandas.Series(0,index=numpy.arange(24))
        # vplants have near inf variable cost. no need to calculate. reported as 0
        if self.kind != 'virtual':
            for period,generated_power in self.online_data.loc[0:23].loc[self.online_data.loc[0:23,'power']>=self.available_min_power.loc[0:23],'power'].items():
                for segment_index in reversed(self.marginal_costs.index):
                    if generated_power >= self.marginal_costs.loc[segment_index,'segment_pmin'] and \
                            generated_power <= self.marginal_costs.loc[segment_index,'segment_pmax']:
                        variable_costs.loc[period] = self.marginal_costs.loc[segment_index,'mw_unit_cost']
                        break
        return variable_costs

    def calculate_one_day_income(self, smp):
        """
        Calculates the system cost to run this plant for the day, considering the online data of the plant at the state it is now
        Uses the linearized mc costs of each bid.
        If the smp is less than the bid, the plant gets the bid price, else the smp price
        Also factors in startup costs
        """
        # first calculate startup cost
        startup_costs = self.calculate_startup_cost()
        # also calculate the energy cost
        energy_costs = 0
        # if we want the income, we only have to multiply the online_data['power'] with the smp and sum
        for period,generated_power in self.online_data.loc[0:23].loc[self.online_data.loc[0:23,'power']>=self.available_min_power.loc[0:23],'power'].items():
            for segment_index in reversed(self.marginal_costs_by_profit_daily.index):
                if generated_power >= self.marginal_costs_by_profit_daily.loc[segment_index,'segment_pmin'] and \
                        generated_power <= self.marginal_costs_by_profit_daily.loc[segment_index,'segment_pmax']:
                    period_cost = max(self.marginal_costs_by_profit_daily.loc[segment_index,'mw_unit_cost'],smp.loc[period])
                    energy_costs += generated_power * period_cost
                    break
        return energy_costs + startup_costs

    def calculate_startup_cost(self):
        startup_costs = 0
        # get the index of all startups
        plant_status_changes = (self.online_data.loc[0:23,'online'] - self.online_data.loc[-1:23,'online'].shift(1)).loc[0:23]
        plant_startups_indices = plant_status_changes.loc[plant_status_changes == 1].index
        # for each startup calculate the cost depending if the plant was hot or cold
        for period in plant_startups_indices:
            if self.online_data.loc[period-1,'downtime'] >= self.tcold:
                startup_costs += self.start_costs[0]
            else:
                startup_costs += self.start_costs[1]
        return startup_costs

    def get_total_produced_power(self):
        """
        Finds  and returns the plants total produced power for the day
        """
        return self.online_data.loc[0:23,'power'].sum()

    def save_online_data(self, day):
        """
        Concatenates the plant's online data to the saved_online_data multiindex dataframe.
        L1 index is the date and L2 the hour, while columns are 'online','power','uptime,'downtime'
        """
        # copy saved online data, reindex and concat with saved_online_data
        current_online_data = self.online_data.loc[0:23,['online','power','uptime','downtime']].copy(deep=True)
        current_online_data = pandas.DataFrame(current_online_data.values,index=pandas.MultiIndex.from_product([[day],current_online_data.index]),columns=['online','power','uptime','downtime'])
        # also add the current cost & profit factor as a new col
        current_online_data = current_online_data.assign(running_cost = self.calculate_one_day_variable_cost(self.online_data.loc[0:23]).tolist())
        current_online_data = current_online_data.assign(profit_factor = self.profit_factor)
        # finally, add the carbon emissions and cost of the unit as new cols before concat
        current_online_data = current_online_data.assign(total_emissions = current_online_data.loc[:,'power'].apply(self.calculate_carbon_cost,args=('total_carbon_emissions',)))
        current_online_data = current_online_data.assign(total_carbon_cost = current_online_data.loc[:,'power'].apply(self.calculate_carbon_cost,'total_carbon_cost'))

        self.saved_online_data = pandas.concat([self.saved_online_data,current_online_data])


    def reset_online_data(self):
        """
        This resets online data for the next uc run, preserving the final uptime/downtime values
        """
        # preserve the final ut/dt values
        self.online_data.loc[-1,['online','power','uptime','downtime','can_recommit']] = [0, 0.0, 0, 0, 1]
        self.online_data.loc[-1,['uptime','downtime']] = [self.online_data.loc[23,'uptime'],self.online_data.loc[23,'downtime']]
        # and reset everything else
        self.online_data.loc[0:,['online','power','uptime','downtime','can_recommit']] = [0, 0.0, 0, 0, 1]

    def reset_extra_costs(self):
        """
        Reset extra costs to be zero, for the next uc solve attempt
        """
        self.special_extra_costs = [0]*24
