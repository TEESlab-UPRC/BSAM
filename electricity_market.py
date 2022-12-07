"""
    Copyright (C) 2022 Technoeconomics of Energy Systems laboratory - University of Piraeus Research Center (TEESlab-UPRC)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
#
"""
This module is the representation of the electricity market.
All the market-specific data can be accessed from here and the model is also running
by calling functions of the market and its sub-parts.
"""

import dataio
import copy
import pandas
import generators
import agents
import numpy
import datetime
import fuel as fuel_module

class ElectricityMarket:
    """
    ElectricityMarket is an object containing an instance of the electricity market, namely:
    The agents owning plants & their plants, each with its respective attributes
    [By definition each single plant agent will own only one plant, but in the future
    multi plant portofolios could be implemented as a combination of single plant agents]
    Load forecasts
    Fuel price forecasts (as a part of the fuel module)
    RES generation forecasts (as a part of the fuel module)
    Various other data such as the reserve margin required (0~1)*100%
    """
    def __init__(self, market_init_data):
        """
        Create the market object by reading the files in the paths specified and creating the
        required agent, generator and fuel objects
        """
        # get the valid period for modelling. This is be used in all functions that depend on
        # knowing what happened in the last years, as this data will not be available for certain years.
        self.valid_years_to_model = market_init_data['valid_years_to_model']
        # import demand and convert time to datetime etc.
        self.demand = dataio.load_dataframe_from_csv(market_init_data['demand_path'], None)
        self.demand['day'] = pandas.to_datetime(self.demand['day'], format='%Y-%m-%d')
        self.demand['hour'] = self.demand['hour'].apply(lambda x: datetime.datetime.strptime(x,'%H:%M:%S').time())
        # import the long-term availabilities of all generators - these are used to identify plants shutting down in the long term
        # this is info for the market to know, not the generator, so it cannot be accessed from inside a plant
        self.generator_long_term_availability = dataio.load_dataframe_from_csv(market_init_data['generator_availability_path'])
        # also import import-specific  data
        self.import_generators_data = dataio.load_dataframe_from_csv(market_init_data['import_plants_data_path'])
        import_generators_long_term_availability = dataio.load_dataframe_from_csv(market_init_data['import_availabilities_path'])
        # and concat import availability together with the generator_long_term_availability
        self.generator_long_term_availability = pandas.concat([self.generator_long_term_availability,import_generators_long_term_availability],axis=1)

        # import misc market data
        self.generators_biomass_cofiring = dataio.load_dataframe_from_csv(market_init_data['generators_biomass_cofiring_path'])
        self.market_data = dataio.load_dataframe_from_csv(market_init_data['market_data_path'])
        # this is the market reserve margin. active plants are always required to be able to cover it as needed
        self.reserve_margins = dataio.load_dataframe_from_csv(market_init_data['reserves_requirements_path'])

        # this is the margin of demand to be covered by fast & expensive backup plants.
        # Those are reserved & unused at UC init & utilized if required to meet MUT/MDT problems
        self.backup_reserves = int(self.market_data.loc['backup_reserves', 'data1'])
        self.virtual_plant_capacity = float(self.market_data.loc['virtual_plant_capacity', 'data1'])
        self.linearized_segments_number = int(self.market_data.loc['linearized_segments_number', 'data1'])
        self.market_price_limit = float(self.market_data.loc['market_price_limit', 'data1'])
        self.plant_closing_data = [int(self.market_data.loc['years_back_to_check_plant_closing_decision','data1']),\
                                    float(self.market_data.loc['closing_down_threshold','data1'])]
        # find the generators and create an agent for each generator
        self.fuel = fuel_module.Fuel(market_init_data['smp_data_path'], \
                                    market_init_data['nat_gas_scenario_path'], \
                                    market_init_data['nat_gas_contribution_path'], \
                                    market_init_data['solar_scenario_path'], \
                                    market_init_data['solar_growth_path'], \
                                    market_init_data['solar_park_percentages_path'], \
                                    market_init_data['wind_onshore_scenario_path'], \
                                    market_init_data['wind_onshore_growth_path'], \
                                    market_init_data['wind_offshore_scenario_path'], \
                                    market_init_data['wind_offshore_growth_path'], \
                                    market_init_data['hydro_generation_path'], \
                                    market_init_data['hydro_plant_max_generation_path'], \
                                    market_init_data['hydro_basins_levels_path'], \
                                    market_init_data['hydro_basins_level_randomization'], \
                                    market_init_data['hydro_basins_path'], \
                                    market_init_data['hydro_price_variables_path'], \
                                    market_init_data['thermal_reference_price_path'], \
                                    market_init_data['res_prices_yearly_data_path'], \
                                    market_init_data['import_prices_path'], \
                                    market_init_data['import_prices_randomization'], \
                                    market_init_data['valid_years_to_model'], \
                                    market_init_data['carbon_price_path'], \
                                    self.market_data)

        generators_list = self.get_generators(dataio.load_dataframe_from_csv(market_init_data['generators_path'], None), market_init_data['enabled_generator_kinds'])

        self.single_plant_agents = self.initialize_agents(market_init_data['agent_actions_data_path'], market_init_data['agent_data_data_path'],
                                generators_list, market_init_data['learning_algorithm'], \
                                market_init_data['lspi_policy_path'], market_init_data['load_learner_policies'], market_init_data['uc_verbosity'])

        # if specified, create an res_exports plant to absorb res production when it exceeds demand - demand*reserve
        # for now the model assumes it has infinite capacity to absorb extra power
        self.res_exports_plant = False
        if market_init_data['use_res_exports_plant']:
            res_exports_plant_data = ['res_exports','exports',0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-100,0,0,0,'-']
            self.res_exports_plant = self.create_generator(res_exports_plant_data,0)

        # find the total hydro capacity of the market
        self.total_hydro_capacity = self.get_total_hydro_capacity()
        # create the uc module
        self.uc_module = market_init_data['uc_module']
        if self.uc_module == 'epl':
            import unitcommit_epl
            self.uc_solver = unitcommit_epl.UnitCommit(self.reserve_margins, self.backup_reserves, self.generator_long_term_availability, market_init_data['uc_verbosity'], market_init_data['use_multiprocessing'])

    def get_generators(self, generators_data, enabled_generator_kinds):
        """
        Read the generators_data and generators_availability tables and create and return the generator objects these files describe
        """
        plants_list = []
        virtual_plant_cost_data = [[0,0,0],0]
        # initially create the generators based on the base data - availability & cost metric will be set by create_generator
        for row in generators_data.itertuples():
            if row.kind in enabled_generator_kinds:
                plant = self.create_generator(row,data_ready=True)
                plants_list.append(plant)
                # save the max costs to use for the virtual plant if needed - we care only about b from the cost params
                if virtual_plant_cost_data[0][1] < plant.cost_funct_params[1]:
                    virtual_plant_cost_data = [plant.cost_funct_params, plant.cost_metric]
        # add import generators if available - we could aggregate their capacities, but leaving it as is
        # allows different availabilities and prices to be applied
        if 'imports' in enabled_generator_kinds:
            for import_generator_name in self.import_generators_data.index:
                capacity = self.import_generators_data.loc[import_generator_name,'capacity']
                cost_metric = self.fuel.import_prices[import_generator_name].mean().mean()
                import_plant_data = [import_generator_name,'imports','imports',capacity,0,0,0,0,0,cost_metric,0,0,0,0,0,0,0,0,0,0,0,0,'-']
                plants_list.append(self.create_generator(import_plant_data,cost_metric))
        # finally add a virtual plant that will be used to cover up problems with meeting the demand
        # this plant will be of kind='virtual', with a max capacity equal to the one specified in the market_data file
        # with startup times = 0 and price equal to the most expensive plant available.
        # it will be used to meet needed demand if this cannot be done any other way
        if 'virtual' in enabled_generator_kinds:
            virtual_plant_data = ['virtual_plant','virtual','virtual',self.virtual_plant_capacity,0,0,0,0,virtual_plant_cost_data[0][0],\
                                virtual_plant_cost_data[0][1],virtual_plant_cost_data[0][2],0,0,0,0,0,0,0,-100,0,0,0,'-']
            plants_list.append(self.create_generator(virtual_plant_data,virtual_plant_cost_data[1]))
        # finally return the plants list
        return plants_list


    def create_generator(self,plant_data,cost_metric=[[0,0,0],0],data_ready=False):
        """
        Creates & returns a generator from the data given
        """
        # if not given all the data we need, ready-made, fix it
        if not data_ready:
            plant_data = pandas.Series(plant_data,index=['Name','kind','fuel','Pmax','Pmin','MUT','MDT','inist','a','b','c','a_hr','b_hr','c_hr','carbon_coefficient','hc','cc','tcold','must_run','ramp_up','ramp_down','factor_minimum_uptime','basin'])
        # and create the plant with the data given
        plant = generators.Generator(plant_data,self.linearized_segments_number,self.market_data)
        # depending on this was a ready_made plant or not, fix the plants cost metric
        if data_ready:
            # if this was a ready_made plant, we can now get the cost metric = cost of the plant at its mid power point
            # remember that the real cost to the market depends on the bidding and will have to be calculated anew
            plant.cost_metric = plant.calculate_cost_metric('cost')
        else:
            plant.cost_metric = cost_metric
        return plant


    def get_one_day_demand(self, day):
        """
        Find and return the demand (load) forecast for the day specified
        """
        demand_data = self.demand.loc[self.demand['day'] == day]
        # find the pv value
        pv_production = self.fuel.pv_scenario.loc[self.fuel.pv_scenario['datetime'] == day].iloc[0]
        # find the two wind values
        wind_production_onshore = self.fuel.wind_onshore_scenario.loc[self.fuel.wind_onshore_scenario['datetime'] == day].iloc[0]
        wind_production_offshore = self.fuel.wind_offshore_scenario.loc[self.fuel.wind_offshore_scenario['datetime'] == day].iloc[0]
        # find the total demand by detracting the imports + wind + pv generation from the load
        demand = pandas.DataFrame(0.0,index=numpy.arange(len(demand_data)),columns=['demand'])
        excess_res_power = [0] * len(demand_data)
        for iperiod in range(len(demand_data)):
            demand_value = round(demand_data.iloc[iperiod,2]-pv_production.iloc[iperiod+1]-wind_production_onshore.iloc[iperiod+1]-wind_production_offshore.iloc[iperiod+1],3)
            # get the amound of demand that must not be covered by RES
            reserves_req = self.reserve_margins.loc[iperiod,'FCR_downwards_min'] + self.reserve_margins.loc[iperiod,'FRR_downwards_min']
            # make certain that the final demand is above the reserve requirement
            if demand_value >= reserves_req:
                demand.iloc[iperiod] = demand_value
            # if final demand is under the reserves requirement due to too much RES, there are many things we could do.
            # eg. route the RES generation to exports, or limit their production, or use a virtual plant to absorb the extra
            # right now, we route the excess production to a virtual "res_exports" plant until we can solve the system via UC
            else:
                print ('Warning! RES production exceeds demand - reserve margin in period %s!' %iperiod)
                print ('...adding spare res power to the res_exports plant')
                # find the amount of power that is not needed, and store it to the excess_power list to be used for reducing RES generation later on
                excess_res_power[iperiod] = round(reserves_req,3) - demand_value
                # also update demand to equal the reserve
                demand.iloc[iperiod] = reserves_req
        if sum(excess_res_power) > 0:
            self.remove_res_generation(day,excess_res_power)
        return round(demand,3)


    def remove_res_generation(self,day,excess_res_power):
        """
        If too much RES hinders our ability to keep sane reserves, we cannot solve the UC problem.
        This routes the RES generation to exports if a virtual "res_exports" plant exists,
        or limits the RES production if not
        """
        # find the pv value
        pv_production = self.fuel.pv_scenario.loc[self.fuel.pv_scenario['datetime'] == day].iloc[0]
        # find the two wind values
        wind_production_onshore = self.fuel.wind_onshore_scenario.loc[self.fuel.wind_onshore_scenario['datetime'] == day].iloc[0]
        wind_production_offshore = self.fuel.wind_offshore_scenario.loc[self.fuel.wind_offshore_scenario['datetime'] == day].iloc[0]

        for period in range(24):
            total_period_res_production = pv_production[period+1] + wind_production_onshore[period+1] + wind_production_offshore[period+1]
            if excess_res_power[period] > 0:
                if total_period_res_production > excess_res_power[period]:
                    print ('Warning! RES exceed demand - reserve margin in period %s!' %period)
                    print ('...adding spare res power to the res_exports plant')
                    # reduce res production proportionally as needed
                    res_pv_share = (pv_production[period+1] / (pv_production[period+1] + wind_production_onshore[period+1] + wind_production_offshore[period+1])) * excess_res_power[period]
                    res_w_onshore_share = (wind_production_onshore[period+1] / (pv_production[period+1] + wind_production_onshore[period+1] + wind_production_offshore[period+1])) * excess_res_power[period]
                    res_w_offshore_share = (wind_production_offshore[period+1] / (pv_production[period+1] + wind_production_onshore[period+1] + wind_production_offshore[period+1])) * excess_res_power[period]
                    hourly_pv_production = pv_production[period+1] - res_pv_share
                    hourly_wind_production_onshore = wind_production_onshore[period+1] - res_w_onshore_share
                    hourly_wind_production_offshore = wind_production_offshore[period+1] - res_w_offshore_share
                    # add the spare res production to the res_exports plant if it exists
                    if isinstance(self.res_exports_plant,generators.Generator):
                        self.res_exports_plant.online_data.loc[period,'power'] += excess_res_power[period]
                    # or reduce the res generation accordingly if it does not
                    else:
                        self.fuel.pv_scenario.loc[self.fuel.pv_scenario['datetime'] == day].iloc[0][period+1] = hourly_pv_production
                        self.fuel.wind_onshore_scenario.loc[self.fuel.wind_onshore_scenario['datetime'] == day].iloc[0][period+1] = hourly_wind_production_onshore
                        self.fuel.wind_offshore_scenario.loc[self.fuel.wind_offshore_scenario['datetime'] == day].iloc[0][period+1] = hourly_wind_production_offshore
                else:
                    print('Error. Not enough RES power to reduce!')
                    import ipdb;ipdb.set_trace()

    def initialize_agents(self, agent_actions_data_path, agent_data_datapath, generators_list, learning_algorithm, \
                            lspi_policy_path, load_learner_policies, verbosity):
        """
        Initialize and return the market agents, each one owning a generator
        """
        current_agents = []
        print ("Creating agents and loading saved agent policies as necessary")
        for plant in generators_list:
            current_agents.append(agents.Single_Plant_Agent(agent_actions_data_path, agent_data_datapath, plant, \
                                self.plant_closing_data, self.demand, learning_algorithm, \
                                lspi_policy_path, load_learner_policies, self.market_data, verbosity))
        return current_agents


    def calculate_res_costs(self,day,curent_smp):
        """
        Calculate and return the cost of res-produced electricity for the day.
        Since solar production is always aggregated, but rooftop produced power differs in price with solar park produced,
        this is handled by using a table with the correct % of each for the current year
        """
        # find the res prices - the res types specified here will be the ones used
        res_prices = self.fuel.calculate_res_prices(day)
        # and get the total costs, by aggregating all given type costs by finding the related res production and multiplying
        total_res_costs = 0
        for res_type in res_prices:
            price = res_prices[res_type]
            costs = 0
            # the price being 'smp' means that it follows the current smp
            if price == 'smp':
                price = curent_smp
            # for solar, the production will be multiplied by the percent produced per type (solar park vs rooftop) to get it's specified cost
            if res_type in ['solar_rooftop','solar_parks']:
                costs = self.fuel.pv_scenario.loc[self.fuel.pv_scenario['datetime'] == day].iloc[0][1:].sum() * self.fuel.solar_percents_per_type[res_type] * price
            elif res_type == 'wind_onshore':
                costs = self.fuel.wind_onshore_scenario.loc[self.fuel.wind_onshore_scenario['datetime'] == day].iloc[0][1:].sum() * self.fuel.solar_percents_per_type[res_type] * price
            elif res_type == 'wind_offshore':
                costs = self.fuel.wind_offshore_scenario.loc[self.fuel.wind_offshore_scenario['datetime'] == day].iloc[0][1:].sum() * self.fuel.solar_percents_per_type[res_type] * price
            total_res_costs += costs
        return total_res_costs


    def get_total_hydro_capacity(self):
        """
        Find and return the total hydro capacity of the market
        """
        total_hydro_capacity = 0
        for agent in self.single_plant_agents:
            if agent.plant.kind == 'hydro':
                total_hydro_capacity += agent.plant.pmax
        return total_hydro_capacity


    def close_down_plants(self,plant,day):
        """
        Handles plants closing down.
        Records the fact within the plant and updates the relevant generator_long_term_availability dataframes
        """
        self.generator_long_term_availability.loc[day.year+1:,agent.plant.name] = 0
        self.uc_solver.generator_long_term_availability.loc[day.year+1:,agent.plant.name] = 0
        plant.close_down = [True,(day+datetime.timedelta(days=1)).date()]

    def initialize_market(self,day,first_day,last_day):
        """
        Run functions that need to be run at the start of the modelling period
        """
        for agent in self.single_plant_agents:
            # update availability (both pmax & pmin) for all plants
            agent.plant.available_power[:] = agent.plant.pmax * self.generator_long_term_availability.loc[day.year,agent.plant.name]
            # cross zonal capacities may be further adjusted using the czc_fmax
            if agent.plant.kind in ['imports']:
                agent.plant.available_power[:] = agent.plant.available_power * self.market_data.loc['cross_zonal_capacity_fmax','data1']
            agent.plant.available_min_power[:] = agent.plant.pmin * self.generator_long_term_availability.loc[day.year,agent.plant.name]
            agent.plant.update_available_pmax_pmin()
            # and update the carbon cost for the year
            agent.plant.carbon_price = self.fuel.carbon_price.loc[day.year][0]

        # get the new water price for all basins if we got hydro plants (and this is not the last day)
        if not last_day and self.total_hydro_capacity > 0:
            # since the monthly updates are done in the last day of the month, add one more day to get the water price for next month
            self.fuel.update_water_price(day+datetime.timedelta(days=1),self.single_plant_agents)

    def do_last_day_updates(self,day,smp):
        """
        Run functions that need to be run at the end of the modelling period
        """
        # since the averages are only updated monthly, update for the last month that passed
        self.fuel.append_monthly_averages(day,self.total_hydro_capacity,last_day=True)
        # and add the latest profit data to the agents
        for agent in self.single_plant_agents:
            profit = agent.calculate_profit(smp)
            # update the taken_actions list of the agent
            agent.taken_actions=agent.taken_actions.append(pandas.DataFrame([[agent.learner.current_action[0],agent.learner.current_action[1],profit]],index=[day],columns=['action','exploit_status','profit']))

    def do_yearly_updates(self,day,last_day):
        """
        Run plant closing down functions and other functions that need only run yearly
        This runs at the last day of each year & typically called by main
        """
        # update the long_term_availabilities of plants & do the closing down of plants that need to close
        for agent in self.single_plant_agents:
            # update availability (both pmax & pmin) for all plants
            agent.plant.available_power[:] = agent.plant.pmax * self.generator_long_term_availability.loc[day.year,agent.plant.name]
            # cross zonal capacities may be further adjusted using the czc_fmax
            if agent.plant.kind in ['imports']:
                agent.plant.available_power[:] = agent.plant.available_power * self.market_data.loc['cross_zonal_capacity_fmax','data1']
            agent.plant.available_min_power[:] = agent.plant.pmin * self.generator_long_term_availability.loc[day.year,agent.plant.name]
            agent.plant.update_available_pmax_pmin()
            # do not do closing down decisions for all virtual plants, but do for everyone else
            if agent.plant.kind != 'virtual' and agent.closing_down_decision(day.date(),self.valid_years_to_model):
                # just update availabilities in both market and UC. Also record the event.
                self.close_down_plants(agent.plant,day)
            # update the carbon cost for the year
            agent.plant.carbon_price = self.fuel.carbon_price.loc[day.year][0]

        # regen the reservoir levels for the next year & calculate prices
        self.fuel.randomize_reservoir_levels()

    def do_monthly_updates(self,day,first_day,last_day):
        """
        Update the fuel price averages lists on each new month with last months data.
        Also find the new water price
        This runs at the last day of each month & typically called by main
        """
        # save & reset the averages list
        self.fuel.append_monthly_averages(day,self.total_hydro_capacity)
        self.fuel.reset_averages()

        # if data exist, also calculate the thermal monthly reference price for the next month
        # such data shall exist one year after start & afterwards
        if day.replace(year=day.year-1,day=1) >= first_day:
            self.fuel.calculate_thermal_monthly_ref_price(day+datetime.timedelta(days=1), self.single_plant_agents)

        # get the new water price for all basins if we got hydro plants (and this is not the last day)
        if not last_day and self.total_hydro_capacity > 0:
            # since the monthly updates are done in the last day of the month, add one more day to get the water price for next month
            self.fuel.update_water_price(day+datetime.timedelta(days=1),self.single_plant_agents)

    def do_daily_updates(self,day,start_day):
        """
        On each new day, update the nat. gas prices and water prices as needed.
        This runs at the start of the day & typically called by calculate_unit_commit
        """
        # get the nat gas price factor and apply it to nat gas plants
        nat_gas_price_factor = self.fuel.get_nat_gas_price_factor(day)
        # update the generators & agents as needed
        for agent in self.single_plant_agents:
            if agent.plant.kind == 'hydro':
                agent.plant.cost_funct_params[1] = self.fuel.get_hydro_generator_price(agent.plant,day,start_day)
            elif agent.plant.kind in ['nat-gas-ccgt','nat-gas-occgt']:
                agent.plant.fuel_factor = nat_gas_price_factor
            elif agent.plant.kind == 'imports':
                # get and set the price for the specific imports generator
                agent.plant.cost_funct_params[1] = self.fuel.get_imports_plant_price(day,agent.plant.name,self.import_generators_data.loc[agent.plant.name,'volatility']/self.market_data.loc['imports_volatility_divisor','data1'])

    def update_fuel_history(self,result,day):
        """
        Update the fuel history.
        Specifically the SMP, nat. gas contribution and hydro generation values of the fuel module.
        This runs at the end of a day (after the unit commit calculation) & is typically called by main
        """
        # get the produced power (total and hydro)
        produced_power = self.fuel.get_produced_power(result[0],{'total':0,'hydro':0})
        # save the hydro data
        self.fuel.daily_hydro_generation.append(produced_power['hydro']/1000)
        # find & save the smp in fuel as calculated in unitcommit (res & res_exports excluded)
        # both daily & hourly averages are saved appropriately
        self.fuel.update_daily_smp(result[2],day)
        # find,save the nat gas contribution (nat-gas-ccgt and nat-gas-occgt aggregated)
        nat_gas_contribution = sum(self.fuel.get_plants_contribution(result[0],produced_power['total'],{'nat-gas-ccgt':0,'nat-gas-occgt':0}).values())
        self.fuel.daily_natural_gas_contribution.append(nat_gas_contribution)
        # find,save the total hydro cost per MWh for the day - avoid divide by zero by setting cost to 0 in case there was no profuced hydro power
        daily_hydro_cost = 0
        if produced_power['hydro'] != 0:
            daily_hydro_cost = self.fuel.get_plants_cost(result[0],{'hydro':0},result[2])['hydro'] / produced_power['hydro']
        self.fuel.daily_hydro_cost.append(daily_hydro_cost)

    def update_solver_data(self,day):
        """
        Update the day demand and the agents list for the solvers to use for the uc
        """
        self.uc_solver.single_plant_agents = self.single_plant_agents
        self.uc_solver.day_demand = self.get_one_day_demand(day)

    def check_for_unfairly_high_prices_and_apply_penalties(self):
        """
        Does a check to make sure that the system marginal price is not unfairly high
        due to agents co-operating to raise the price.
        In such occasion, a penalty is applied to their profit
        """
        # first get the needed data. This is for each agent a tuple (power_produced,action_taken)
        # this data includes the agent action, the power produced and the total power produced
        agents_data = []
        total_power_produced = 0
        for agent_index,agent in enumerate(self.single_plant_agents):
            # pure peakers are expempted, and also agents that do not decide on bid price.
            if agent.plant.kind not in ['nat-gas-occgt','imports','hydro']:
                plant_power_produced = agent.plant.online_data.loc[0:23,'power'].sum()
                total_power_produced += plant_power_produced
                # get the action chosen
                agents_data.append((agent.actions_list[agent.learner.current_action[0]],plant_power_produced,agent_index))

        # then find out if some agents are raising the price all together
        aggregated_profit_margin = 0

        for data in agents_data:
            if total_power_produced != 0:
                aggregated_profit_margin += data[0] * data[1]/total_power_produced

        if aggregated_profit_margin > self.market_price_limit:
            # we need to act. apply an exponential penalty to all the agents above the limit
            print ('Plant profits above limit. Applying penalties')
            for data in agents_data:
                if data[0] >= self.market_price_limit:
                    # this is the % penatly to be applied to profits
                    # it should start at the market_price_limit and increase by 10% for each 0.05 step
                    penalty_percent = 100*(data[0]-(self.market_price_limit-0.05))/0.5
                    self.single_plant_agents[data[2]].profit_penalty = penalty_percent

    def calculate_unit_commit(self,day,start_day):
        """
        Calculate the UC for the day specified
        """
        # 1. do the updates using the last days data
        self.do_daily_updates(day,start_day)
        # reset the plants online_data for the next uc run if it is not the start date (where everything is already correct)
        self.reset_agents_data(day,start_day)
        # update the uc attributes as needed before starting the uc
        self.update_solver_data(day)
        # run the uc
        result = self.uc_solver.calculate_unit_commit(day)
        if sum(result[3]) > 0:
            # in this case, the problem had too few demand, possibly due to too much RES. remove required excess RES power
            self.remove_res_generation(day,result[3])
        # check that demand was correctly met
        # get generation first
        daily_generation = pandas.Series(0,index=self.single_plant_agents[0].plant.online_data.loc[0:23].index,name='power',dtype='float64')
        # get power of conventional (and hydro) plants. do not add RES as it was subtracted from demand
        for agent in self.single_plant_agents:
            daily_generation += agent.plant.online_data.loc[0:23,'power']
        daily_demand = self.uc_solver.day_demand.loc[:,'demand']
        # also test with RES
        # find the pv value
        pv_production = self.fuel.pv_scenario.loc[self.fuel.pv_scenario['datetime'] == day].iloc[0].drop('datetime').sum()
        # find the two wind values
        wind_production_onshore = self.fuel.wind_onshore_scenario.loc[self.fuel.wind_onshore_scenario['datetime'] == day].iloc[0].drop('datetime').sum()
        wind_production_offshore = self.fuel.wind_offshore_scenario.loc[self.fuel.wind_offshore_scenario['datetime'] == day].iloc[0].drop('datetime').sum()
        # remove any RES exports from res
        res_exports = self.res_exports_plant.online_data.loc[0:23,'power'].sum()
        res_gen = pv_production+wind_production_onshore+wind_production_offshore-res_exports
        total_demand = self.demand.loc[self.demand['day'] == day].demand.sum()

        if not numpy.isclose(daily_generation.sum(),daily_demand.sum()):
            print ('Error. Demand and Generation (w/out RES do not match! Dropping to debug')
            #import ipdb;ipdb.set_trace()

        if not numpy.isclose(daily_generation.sum()+res_gen,total_demand):
            print ('Error. Demand and Generation (w/RES) do not match! Dropping to debug')
            #import ipdb;ipdb.set_trace()

        # if the result is a str it means that there was an error somewhere
        if type(result) is not str:
            # do a check to make sure that the price is not unfairly high
            # due to agents co-operating to raise the price
            # on that occasion, apply a penalty to their profit
            self.check_for_unfairly_high_prices_and_apply_penalties()
            # add the res costs to the final cost
            res_costs = self.calculate_res_costs(day,result[1])
            result = (result[0], result[1] + res_costs, result[2])
            # save the online data for the day
            for agent in self.single_plant_agents:
                agent.plant.save_online_data(day)
            # also save the online data of the res_exports_plant
            if isinstance(self.res_exports_plant,generators.Generator):
                self.res_exports_plant.save_online_data(day)
        # if none of the above was true, this returns the error as a str
        return result

    def get_state(self,day):
        """
        Analyze the uc result to get and return the current state
        as a tuple (total_demand, natural_gas_price, marginal_cost)
        Be careful! If we change the state dimensions, record it in the state_dimensions variable in lspi.csv & also in other leanining modules
        """
        # get the total demand for the day - it is the sum of all demands
        total_demand = self.demand.loc[self.demand['day'] == day].demand.sum()
        # get the Natural Gas fuel price
        natural_gas_price = self.fuel.nat_gas_scenario.nat_gas_price[datetime.datetime(day.year,day.month,1)]
        # get the daily smp
        daily_smp = self.fuel.calculated_smp.loc[day,'daily_smp']
        # make sure that the state has no nans os infs
        if numpy.isnan([total_demand, natural_gas_price, daily_smp]).any() or numpy.isinf([total_demand, natural_gas_price, daily_smp]).any():
            print('A part of the daily state is either inf or nan. Starting debugger')
            import ipdb;ipdb.set_trace()
        # and return the state
        return (total_demand, natural_gas_price, daily_smp)

    def reset_agents_data(self,day,start_day):
        """
        Resets the online data of all plants from all agents to 0, so that the "saved online data" list will be entered the correct values if the plants availability excludes it from uc.
        It also resets the online data of the res_exports_plant if it exists and also resets special_extra_costs thet were applied
        This runs before starting a new modelling day
        """
        for agent in self.single_plant_agents:
            # only reset online data if not just starting
            if day.date() != start_day.date():
                agent.plant.reset_online_data()
            agent.plant.reset_extra_costs()
        if isinstance(self.res_exports_plant,generators.Generator) and day.date() != start_day.date:
            self.res_exports_plant.reset_online_data()
