# -*- coding: utf-8 -*-

"""
This module is used for containing all fuel related data and doing the needed calculations
"""

import dataio
import numpy
import pandas
import datetime
import calendar
import ipdb

class Fuel:
    """
    The Fuel class is always belonging to a market class and is used for accessing all
    fuel-related data and doing the relevant calculations
    """

    def __init__(self, smp_data_path, \
                    nat_gas_scenario_path, \
                    nat_gas_contribution_path, \
                    pv_scenario_path, \
                    solar_growth_path, \
                    solar_park_percentages_path, \
                    wind_onshore_scenario_path, \
                    wind_onshore_growth_path,\
                    wind_offshore_scenario_path, \
                    wind_offshore_growth_path,
                    hydro_generation_path, \
                    hydro_plant_max_generation_path, \
                    hydro_basins_levels_path, \
                    hydro_basins_level_randomization, \
                    hydro_basins_path, \
                    hydro_price_variables_path, \
                    thermal_reference_price_path, \
                    res_prices_yearly_data_path, \
                    import_prices_path, \
                    import_prices_randomization, \
                    valid_years_to_model, \
                    carbon_price_path, \
                    market_data):

        """
        At init, load the input files to create the main class attributes
        """
        self.valid_years_to_model = valid_years_to_model
        # import the smp scenario and convert its index to datetime
        self.smp_data_path = smp_data_path
        self.smp_data = dataio.load_dataframe_from_csv(self.smp_data_path)
        self.smp_data.index = pandas.to_datetime(self.smp_data.index, format='%Y-%m-%d')
        # same for nat_gas prices
        self.nat_gas_scenario = dataio.load_dataframe_from_csv(nat_gas_scenario_path)
        self.nat_gas_scenario.index = pandas.to_datetime(self.nat_gas_scenario.index, format='%Y-%m-%d')
        # same for nat_gas contribution - this is in percents (0~1) and not in (0~100)
        self.nat_gas_contribution_path = nat_gas_contribution_path
        self.nat_gas_contribution = dataio.load_dataframe_from_csv(self.nat_gas_contribution_path)
        self.nat_gas_contribution.index = pandas.to_datetime(self.nat_gas_contribution.index, format='%Y-%m-%d')
        # same for pv production
        self.pv_scenario = dataio.load_dataframe_from_csv(pv_scenario_path,None)
        self.pv_scenario['datetime'] = pandas.to_datetime(self.pv_scenario['datetime'], format='%Y-%m-%d')
        self.pv_growth = dataio.load_dataframe_from_csv(solar_growth_path)
        # solar_percents_per_type describe how much of the solar energy is from rooftops vs solar parks
        self.solar_percents_per_type = dataio.load_dataframe_from_csv(solar_park_percentages_path)
        # same for wind production
        # onshore
        self.wind_onshore_scenario = dataio.load_dataframe_from_csv(wind_onshore_scenario_path,None)
        self.wind_onshore_scenario['datetime'] = pandas.to_datetime(self.wind_onshore_scenario['datetime'], format='%Y-%m-%d')
        self.wind_onshore_growth = dataio.load_dataframe_from_csv(wind_onshore_growth_path)
        # and offshore
        self.wind_offshore_scenario = dataio.load_dataframe_from_csv(wind_offshore_scenario_path,None)
        self.wind_offshore_scenario['datetime'] = pandas.to_datetime(self.wind_offshore_scenario['datetime'], format='%Y-%m-%d')
        self.wind_offshore_growth = dataio.load_dataframe_from_csv(wind_offshore_growth_path)
        # same for import prices - load all available data for all import 'plants'
        self.import_prices_randomization = int(import_prices_randomization)
        self.import_prices = {}
        for imports_plant in import_prices_path:
            imports_data = dataio.load_dataframe_from_csv(import_prices_path[imports_plant],None)
            imports_data.index = pandas.to_datetime(imports_data['datetime'], format='%Y-%m-%d')
            imports_data.drop('datetime',axis=1,inplace=True)
            self.import_prices.update({imports_plant.strip('_path'):imports_data})
        # same for co2 prices
        self.carbon_price = dataio.load_dataframe_from_csv(carbon_price_path)
        # same for hydro generation
        self.hydro_generation = dataio.load_dataframe_from_csv(hydro_generation_path)
        self.hydro_generation.index = pandas.to_datetime(self.hydro_generation.index, format='%Y-%m-%d')
        # import reservoir and hydro generation related data
        self.hydro_basins_level_randomization = int(hydro_basins_level_randomization)
        self.hydro_plant_max_generation = dataio.load_dataframe_from_csv(hydro_plant_max_generation_path).T
        self.hydro_basins_levels_initial = dataio.load_dataframe_from_csv(hydro_basins_levels_path)
        self.hydro_basins = dataio.load_dataframe_from_csv(hydro_basins_path).index.tolist()
        # import thermal_reference_prices (cth) for calculation of water price
        self.thermal_reference_price = dataio.load_dataframe_from_csv(thermal_reference_price_path)
        self.thermal_reference_price.index = pandas.to_datetime(self.thermal_reference_price.index, format='%Y-%m-%d')

        # import hydro price calculation variables
        self.hydro_price_variables = dataio.load_dataframe_from_csv(hydro_price_variables_path)
        # also import the res prices scenario
        self.res_yearly_prices = dataio.load_dataframe_from_csv(res_prices_yearly_data_path)
        # import the 'market_data' to use market & misc variables
        self.market_data = market_data
        # store the calculated smp in this dataframe. index will be datetime & columns will be hourly & daily smp
        self.calculated_smp = pandas.DataFrame(columns=['hourly_smp','daily_smp'])
        # these are lists used to store the daily market data in order to use them for calculations (eg. in water price)
        self.daily_natural_gas_contribution = []
        self.daily_hydro_generation = []
        self.daily_hydro_cost = []
        # the water price will get calculated monthly for each basin
        # make a dataframe with the basins as columns and the day as index. initially this will be empty
        self.water_price = pandas.DataFrame(columns=self.hydro_basins)
        # also add an actual_general_price col to save the price of water weighted by the individual plants cost and production.
        # this value will be entered at the end of each month representing the weighted average price for that month
        self.water_price = self.water_price.assign(actual_general_price=None)
        # get the basins from the water price index
        self.basins = self.water_price.columns.tolist()
        self.basins.remove('actual_general_price')
        # Use this dataframe to apply randomizing into the reservoir levels. Generate new levels at program start and yearly
        # This dataframe is used for all reservoir level calculations and has a per-basin and per-month resolution
        self.hydro_basins_levels = self.randomize_reservoir_levels()

    def get_produced_power(self,generator_set,produced_power_dict):
        """
        Parses plants in the generator_set to calculate the total power produced by them.
        The produced_power_dict is a dictionary with specific kinds of plants to aggregate power from.
        The keys can be either a plant kind, or 'total' and their values assumed to be 0 initially
        The produced_power_dict is filled with aggregated produced power from plants.online data and returned
        """
        for plant in generator_set:
            if 'total' in produced_power_dict:
                produced_power_dict['total'] += plant.get_total_produced_power()
            if plant.kind in produced_power_dict:
                produced_power_dict[plant.kind] += plant.get_total_produced_power()
        return produced_power_dict

    def get_plants_contribution(self, generator_set, total_power_produced, plants_contribution_dict):
        """
        Calculates and returns the % of contribution of specified plant kinds from a specific generator_set
        Updates the kinds_dict with the %s
        """
        # get the produced power for each of the kinds specified
        for plant in generator_set:
            if plant.kind in plants_contribution_dict:
                plants_contribution_dict[plant.kind] += plant.get_total_produced_power()
        # convert the produced power values to % contribution
        for key in plants_contribution_dict:
            plants_contribution_dict[key] = plants_contribution_dict[key]/total_power_produced
        return plants_contribution_dict

    def get_plants_cost(self,generator_set,costs_dict,electricity_price):
        """
        Calculates and returns the total cost of all plants specified as keys in the costs_dict using the generator_set
        """
        for plant in generator_set:
            if plant.kind in costs_dict:
                costs_dict[plant.kind] += plant.calculate_one_day_income(electricity_price)
        return costs_dict

    def get_nat_gas_price_factor(self, day):
        """
        Calculates and returns the % to apply to plants cost functions as a result of the
        nat. gas price calculations.
        The current E/MWh values where calculated with a price normalization at 0.3308,
        thus lower/higher prices will have a proportional effect on the final costs
        """
        day_to_check = datetime.datetime(day.year,day.month,1)
        price = self.nat_gas_scenario.nat_gas_price[day_to_check]
        return price/self.market_data.loc['nat_gas_price_factor','data1']

    def calculate_res_prices(self, day):
        """
        Calculates and returns the res prices for the day
        """
        # first find the electricity cost
        # then, calculate the price for this res type
        # if the price is 0, this means that the generated energy is sold using the day's smp - no need to act
        # if the price is lower than 0, then the absolute of the price given is tied to last years system MC as a multiplicative factor
        # get the last_years_marginal_price to use later on
        last_years_marginal_price =  numpy.average(self.smp_data[-12:])
        res_costs = {}
        for res_type in self.res_yearly_prices.columns:
            price = self.res_yearly_prices.loc[day.year,res_type]
            if type(price) is not str and price < 0:
                price = abs(price) * last_years_marginal_price
            res_costs.update({res_type:price})
        return res_costs

    def append_monthly_averages(self,day,total_hydro_capacity,last_day=False):
        """
        Runs at the end of each month and calculates and appends the monthly averages for
        smp, nat. gas contribution and daily hydro generation to the relevant monthly class attribute
        """
        # find the real averages for this month
        natural_gas_contribution_ave = 0
        if len(self.daily_natural_gas_contribution) > 0:
            natural_gas_contribution_ave = numpy.average(numpy.array(self.daily_natural_gas_contribution))
        daily_hydro_generation_ave = 0
        if len(self.daily_hydro_generation) > 0:
            daily_hydro_generation_ave = numpy.average(numpy.array(self.daily_hydro_generation))
        # append the real averages to the monthly data
        current_date = datetime.datetime(day.year,day.month,1)
        self.smp_data.smp[current_date] = self.calculated_smp.loc[day,'daily_smp']
        self.nat_gas_contribution.loc[current_date,'natural_gas_contribution'] = natural_gas_contribution_ave
        # ommit hydro if not used
        if total_hydro_capacity > 0:
            self.hydro_generation.loc[current_date,'hydro_generation'] = daily_hydro_generation_ave
            # and also save the general water price for the month - if there was no hydro, just save nan
            if numpy.array(self.daily_hydro_generation).sum() == 0:
                water_price = numpy.nan
            else:
                water_price = (numpy.array(self.daily_hydro_cost) * numpy.array(self.daily_hydro_generation)).sum() / numpy.array(self.daily_hydro_generation).sum()
            if last_day:
                current_date += datetime.timedelta(days=1)
            self.water_price.loc[current_date,'actual_general_price'] = water_price

    def update_daily_smp(self, smp, day):
        # also save the smp in self.system_marginal_costs (using the hourly & daily values)
        daily_smp = numpy.average(smp)
        for period in smp.index:
            smp_data = [smp.loc[period],daily_smp]
            current_datetime = datetime.datetime(day.year,day.month,day.day,period)
            self.calculated_smp = pandas.concat([self.calculated_smp, pandas.DataFrame([smp_data],index=[current_datetime],columns=self.calculated_smp.columns)])

    def reset_averages(self):
        """
        Resets the daily lists for smp, nat. gas contribution and daily hydro generation
        Runs once each month
        """
        self.daily_natural_gas_contribution = []
        self.daily_hydro_generation = []
        self.daily_hydro_cost = []

    def calculate_thermal_monthly_ref_price(self, day, agents):
        """
        Returns the monthly average of the hourly variable cost of the thermal plants, for the same month of last year,
        normalized to the total thermal generation (Cth)
        """
        thermal_reference_price = 0
        periods = 0
        for month_day in range (calendar.monthrange(day.year-1,day.month)[1]):
            current_day_string = datetime.datetime(day.year-1,day.month,month_day+1).strftime("%Y-%m-%d")
            for period in range(24):
                generation_cost = 0
                total_generation = 0
                periods += 1
                for agent in agents:
                    if agent.plant.kind in ['lignite-st','nat-gas-st','nat-gas-ccgt','nat-gas-ocgt']:
                        total_generation += agent.plant.saved_online_data.loc[current_day_string,period].loc['power']
                        generation_cost += agent.plant.saved_online_data.loc[current_day_string,period].loc['power']*agent.plant.saved_online_data.loc[current_day_string,period].loc['running_cost']
                # only add calculate price if generation is other than zero
                if total_generation != 0:
                    thermal_reference_price += generation_cost / total_generation
                else:
                    # if generation from conventionals was zero, do not consider this period for the substitution price calculation
                    periods -= 1

        thermal_reference_price = thermal_reference_price / periods
        # and save cth
        self.thermal_reference_price.loc[day.replace(day=1),'cth'] = thermal_reference_price

    def get_water_price_thermal(self, day):
        """
        Calculates and returns the thermal part of the water value (C1), that depends on the thermal generation displaced.
        Information on the how the calculation is done is provided by the 207/2016 RAE regulatory decision
        """
        # get cth
        cth = self.thermal_reference_price.loc[day.replace(year=day.year-1,day=1),'cth']
        # find the % change of the nat gas price from last year. the price of the other fuels is assumed constant
        nat_gas_price_change_percent = self.nat_gas_scenario.loc[day.replace(day=1),'nat_gas_price'] / \
                                self.nat_gas_scenario.loc[day.replace(day=1).replace(year=day.year-1),'nat_gas_price']
        # get the contribution of nat gas to the mix on this month last year
        nat_gas_contribution = self.nat_gas_contribution.loc[day.replace(day=1),'natural_gas_contribution']
        # calculate the 'sigma' coefficient
        sigma = nat_gas_price_change_percent * nat_gas_contribution
        water_marginal_system_price_thermal_factor = (1+sigma)*cth
        return water_marginal_system_price_thermal_factor

    def get_water_price_reservoir(self,day,water_thermal_price,basin):
        """
        Calculates and returns the reservoir part of the water value (C2),
        that depends on the water reservoir levels (r)
        This is based on basin-specific data loaded within fuel.py
        Information on the calculation can be provided by the paper named:
        "Μεθοδολογία υπολογισμού Μεταβλητού Κόστους των Υδροηλεκτρικών Μονάδων,
        κατά τις διατάξεις του Άρθρου 44, παρ. 5, του ΚΣΗΕ." provided by LAGIE (2013).

        In addition, a short description of the methodology is provided below:
        # C1 is the water_thermal_price
        # C1 is the water_reservoir_price
        # C2 = VCmax - C1 when Rmin > r
        # C2 = (VCmax - C1) *  exp( -k1 * (r-Rmin)/(Rref-Rmin)) when Rref >r > Rmin
        # C2 = 0 when r in [Rrefdn, Rrefup]
        # C2 = - C1 * exp( -k2 * (Rmax - r)/(Rmax-Rref)) when Rref < r <Rmax
        # C2 = -C1 when r2 > Rmax
        # where:
        # Rmax & Rmin are the max&min reservoir levels for the last 10 years
        # Rref is the mean value of the reservoir levels for the last 10 years
        # Rrefup = Rref + TOLup (Rmax-Rref)
        # Rrefdn = Rref - TOLdn (Rref-Rmin)
        # if r > Rsec > Rmax, the plant can inject power for free in order to avoid a flood
        # TOLup/TOLdn are %
        # VCmax (μέγιστο κατώφλι Προσφορών Έγχυσης Υδροηλεκτρικών Σταθμών) < Διοικητικά Οριζόμενης Μέγιστης Τιμής Προσφορών Έγχυσης (150E/MWH)
        # k1/k2 rythmos metavolis  timis
        # k1 = -(Rref-Rmin)/(r-Rmin) * ln(C2/(VCmax-C1). eg. 40% *(VCmax-C1) increase when reservoir is 30% of Rref-Rmin: k1 = 3.054
        # k2 = -(Rmax-Rref)/(Rmax-r2) * ln (C1-VC)/C1. eg. decrease to 60%*C1 when res is 70% of Rmax-Rref : k2 = -ln(0.4)/0.3 = 3.054
        """
        water_reservoir_price = 0
        vcmax = self.hydro_price_variables.value.loc['vcmax']

        rmax = self.hydro_basins_levels.loc[day.month,basin+'_rmax']
        rmin = self.hydro_basins_levels.loc[day.month,basin+'_rmin']
        rref = self.hydro_basins_levels.loc[day.month,basin+'_rref']

        rrefup = rref + self.hydro_price_variables.value.loc['tolup'] * (rmax-rref)
        rrefdn = rref - self.hydro_price_variables.value.loc['toldn'] * (rref-rmin)
        k1 = self.hydro_price_variables.value.loc['k1']
        k2 = self.hydro_price_variables.value.loc['k2']
        # get the water level in the reservoir & the delta of the available water to the ref water

        reservoir_level = self.hydro_basins_levels.loc[day.month,basin+'_water_lvl']
        # if the reservoir level has less or equal water to the rmin, price is C2 = VCmax-C1
        if reservoir_level <= rmin:
            water_reservoir_price = vcmax - water_thermal_price
        # if we are over rmin but under rrefdn, value is calculated as (VCmax - C1) *  exp( -k1 * (r-Rmin)/(Rref-Rmin)) when Rref >r > Rmin
        elif rrefdn > reservoir_level > rmin:
            water_reservoir_price = (vcmax - water_thermal_price) * numpy.exp( -k1 * (reservoir_level-rmin)/(rref-rmin))
        # if the water level is in [rrefdn,rrefup], C2=0
        elif rrefup >= reservoir_level >= rrefdn:
            water_reservoir_price = 0
        # if we are over rrefup but under rmax, C2 = - C1 * exp( -k2 * (Rmax - r)/(Rmax-Rref))
        elif rmax > reservoir_level > rrefup:
            water_reservoir_price = -water_thermal_price * numpy.exp(-k2 * (rmax - reservoir_level)/(rmax-rref))
        # finally if we are at rmax, C2 = -C1 (water value total = 0)
        elif reservoir_level >= rmax:
            water_reservoir_price = -water_thermal_price
        return water_reservoir_price

    def update_water_price(self, day, market_agents):
        """
        Calculates and returns the water value as a sum of the thermal and the reservoir parts of the water price.
        This must run only once per month or index conflicts will occur
        """
        # create a dataframe to store the new prices - initially filled with nans
        water_prices = pandas.DataFrame(columns=self.water_price.columns,index=[datetime.datetime(day.year,day.month,1)])
        # get the thermal substitution price part of the price
        water_thermal_price = self.get_water_price_thermal(day)
        # for each of the known river basins, calculate the price
        for basin in self.basins:
            water_price = 0
            water_reservoir_price = self.get_water_price_reservoir(day,water_thermal_price,basin)
            water_price = water_reservoir_price+water_thermal_price
            water_prices.loc[datetime.datetime(day.year,day.month,1),basin]=water_price
        # and finally update the self.water_price df by concatting with water_prices
        self.water_price=pandas.concat([self.water_price,water_prices])

    def randomize_reservoir_levels(self):
        """
        Using the self.hydro_basins_levels_initial dataframe for index and columns,
        generate a reservoir level height by adding the product of a a normal distribution
        with mean equal to 0 and st.dev equal for each basin to the one in self.hydro_basins_levels_initial['volatility']
        The modifiers are yearly, since we assume that each basin is affected on a per-year basis.
        If hydro_basins_level_randomization is 0, this is deactivated and the mean values returned instead
        """
        new_reservoir_levels = pandas.DataFrame(self.hydro_basins_levels_initial.values,index=self.hydro_basins_levels_initial.index,columns=self.hydro_basins_levels_initial.columns)
        if self.hydro_basins_level_randomization > 0:
            for basin in self.basins:
                for month in self.hydro_basins_levels_initial.index:
                    modifier = numpy.random.normal(0,self.hydro_basins_levels_initial.loc[month,basin+'_water_lvl_volatility']/2) # underestimate by 50% to smoothen the changes
                    new_reservoir_levels.loc[month,basin+'_water_lvl'] = (self.hydro_basins_levels_initial.loc[month,basin+'_water_lvl'] + modifier).round()
        return new_reservoir_levels

    def get_hydro_generator_price(self,plant,day,start_day):
        """
        Get the daily price for a hydro generator, depending on the energy it has produced
        up to now and the price indicated by the stats for the basin it belongs to.
        Since the expected generation stats are monthly but the price is modified daily,
        expected generation is handled proportionally for the last and initial months
        """
        # no bidding is involved for hydro plants, but their marginal cost is dependent on the monthly water price - get this price
        price = self.water_price.loc[datetime.datetime(day.year,day.month,1),plant.basin]
        # fix the starting day for the year we are in. this will be the 1/1/xxxx if it is not the first year.
        if start_day.year != day.year:
            start_day = datetime.datetime(day.year,1,1)

        # get the proportion of the first month that will be modelled & the proportion of the last month currently modelled.
        # to find them first handle end-of year situations for next months
        if start_day.month < 12:
            start_second_month = datetime.datetime(start_day.year,start_day.month+1,1)
        else: start_second_month = datetime.datetime(start_day.year+1,1,1)

        if day.month < 12:
            start_next_month = datetime.datetime(day.year,day.month+1,1)
        else: start_next_month = datetime.datetime(start_day.year+1,1,1)
        start_this_month = datetime.datetime(day.year,day.month,1)
        # and then just calculate them by using datetime timedeltas
        proportion_of_initial_month_to_model = (start_second_month - start_day).days / (start_second_month - datetime.datetime(start_day.year,start_day.month,1)).days

        # if we are at the initial month, the current modelled period will be different since we may have not started at the beginning
        if day.month == start_day.month:
            proportion_of_current_month_modelled = (day - start_day).days / (start_next_month - start_day).days
        # else just get the whole month up to now
        else:
            proportion_of_current_month_modelled = (day - start_this_month).days / (start_next_month - start_this_month).days
        #if proportion_of_current_month_modelled >0 and day.month != start_day.month: import ipdb;ipdb.set_trace()
        # find the expected generation up to this point. aggregate initial and final months to everything else
        # start at the expected gen for the current month
        expected_generation = self.hydro_plant_max_generation.loc[str(day.month),plant.name] * proportion_of_current_month_modelled
        # if this is the only month right now stop here, else add the generation for the initial month & middle months as needed
        if day.month != start_day.month:
            # the initial months generation will equal the proportion_of_initial_month_to_model * expected gen in that month
            expected_generation += self.hydro_plant_max_generation.loc[str(start_day.month),plant.name] * proportion_of_initial_month_to_model
            # add the expected gen for all months up to the last if this is not the first or last month
            if day.month > start_day.month + 1:
                expected_generation += self.hydro_plant_max_generation.loc[str(start_day.month+1):str(day.month-1), plant.name].sum()

        # also get the current generation, easily by summing the generation for the current modelling period
        current_generation = 0
        if not plant.saved_online_data.empty:
            current_generation = plant.saved_online_data.loc[start_day:day,'power'].sum()

        # crossreference the current yearly generation with the expected yearly generation up to this point (which represents available water energy) and modify the price as needed
        # find the total energy difference & save it to the plant
        total_difference = expected_generation - current_generation
        plant.produced_energy_difference = total_difference
        # apply a change to the price proportional to the total difference
        price_mod = 1.0

        # a lack of water will just drive the price way up, while a surplus will drive it down but more slowly
        # if we already passed the max energy available, increase prices up to the allowed max
        # the price increases exponentially, with the exponent specified in other.csv
        if total_difference < 0:
            price_mod = pow((1.0 + abs(total_difference)/expected_generation),float(self.market_data.loc['hydro_price_mod_increase_exponent','data1']))

        # if we got energy to spare, lower prices but only if the difference is higher than the expected for some days
        # (specified externally as days_to_wait_before_lowering_hydro_price). This means that the
        # difference must be higher that the average expected production within this month * days_to_wait_before_lowering_hydro_price - this should make this lag a bit
        elif total_difference > float(self.market_data.loc['days_to_wait_before_lowering_hydro_price','data1']) * \
                                        self.hydro_plant_max_generation.loc[str(day.month),plant.name] / \
                                        (start_next_month - start_this_month).days:
            price_mod = 1.0 - pow((total_difference / expected_generation),2)

        # apply the mod to the price and return. take care to not prices over the max allowed
        price = min(self.hydro_price_variables.value.loc['vcmax'], price * price_mod)
        #print (plant.name,'price:',price,'expected:',expected_generation,'current:',current_generation,'total diff:',total_difference)
        #print (price)
        return price

    def get_imports_plant_price(self,day,plant_name,volatility):
        """
        Returns a specific cost for the imports plant & day specified
        This uses the imports_price as a basis but adhers to a normal distribution with a std. dev already given
        The randomization is not applied if the import_prices_randomization variable is 0
        """
        # get the mean price of imports for the next day.
        imports_price = self.import_prices[plant_name].loc[day].mean()
        # shift this mean price if needed. we will use this for assigning costs for imports plants
        imports_price *= (1 + self.market_data.loc['imports_price_shift_percent','data1'])
        # save this price as the imports price for the day, overwriting the old value
        self.import_prices[plant_name].loc[day] = imports_price
        if self.import_prices_randomization > 0:
            # further modify the price via a normal distribution modifier and make sure its lower than the system max price and return
            imports_price = min(self.market_data.loc['max_electricity_price','data1'], numpy.random.normal(imports_price, volatility))
        return imports_price
