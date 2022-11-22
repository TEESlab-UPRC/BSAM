
#!/bin/python
"""
This is the main module of the Wholesale Market Sim
The purpose of the sim is to model the complexities of a wholesale electricity market,
being able to specifically model the agent begaviour, bidding procedure and
unit commitment/market clearence processes correctly, while also taking into account
a wide range of data such as load demand, fuel prices and res generation forecasts.
"""

import electricity_market
import pandas
import numpy
import datetime
import dataio
import data_extractor
import configobj
import sys
import copy
import importlib
import multiprocessing
import warnings

# modify the systems max recursion limit to fix a problem when pickling all the data into a single file caused by pickling keras models
sys.setrecursionlimit(10000)
class Main:
    """
    This is the main class of the sim.
    At init it first loads the .ini file specified and applies any overrides given at creation.
    Then it creates the market, based on the datapaths given & uses this market object to do the calculations needed for the sim to work.
    This class can also draw upon the matplotlib library to visualize generated results.
    """

    def __init__(self, init_file = 'data/greece_baseline_ipto.ini', result_save_path = False, override_program_options={}):
        # first read the ini file and do any proccessing needed upon it
        self.config = configobj.ConfigObj(init_file)
        # if there are overrides, apply them by updating the config dict
        # these overrides are only allowed for the program_options section
        self.config['program_options'].update(override_program_options)
        # fix the all paths within config by joining the paths specified with the root path
        self.config.walk(dataio.join_path_with_root_folder,config=self.config)
        # convert config str data to numeric/bool as needed
        # convert program options to list of ints rather than a str
        self.config['program_options']['valid_years_to_model'] = list(map(int, self.config['program_options']['valid_years_to_model']))
        # also convert program options to int
        self.config['program_options']['temp_save_interval'] = int(self.config['program_options']['temp_save_interval'])
        # convert program options to bool
        self.config['program_options']['use_multiprocessing'] = bool(int(self.config['program_options']['use_multiprocessing']))
        self.config['program_options']['load_learner_policies'] = bool(int(self.config['program_options']['load_learner_policies']))
        self.config['program_options']['save_learner_policies'] = bool(int(self.config['program_options']['save_learner_policies']))
        self.config['program_options']['clear_lspi_policies'] = bool(int(self.config['program_options']['clear_lspi_policies']))
        self.config['program_options']['save_results'] = bool(int(self.config['program_options']['save_results']))
        self.config['program_options']['use_res_exports_plant'] = bool(int(self.config['program_options']['use_res_exports_plant']))
        self.config['program_options']['verification_mode'] = bool(int(self.config['program_options']['verification_mode']))

        # now config is correct, continue initializing
        # this deletes any saved learner policies if required (and subequently sets load policies to false)
        if self.config['program_options']['clear_learner_policies']:
            if self.config['program_options']['learning_algorithm'] == 'lspi':
                dataio.delete_files_in_folder(self.config['agents_module_datapaths']['lspi_policy_path'])
            self.config['program_options']['load_learner_policies'] = False

        # determine the exact scenario to be able to create the correct market - only one scenario per market instance allowed
        self.scenario_data = {}
        market_init_data = self.choose_scenario_files()


        # this disables learning randomness - to be used for verification purposes so that two runs can be identical
        if self.config['program_options']['verification_mode']:
            # set learner to none
            self.config['program_options']['learning_algorithm'] = 'none'
            # and update scenario data
            market_init_data.update({'learning_algorithm':self.config['program_options']['learning_algorithm']})

        # load the correct cost_benefits_calculator module & data
        global cost_benefits_calculator
        cost_benefits_calculator = importlib.import_module(self.config['scenario_datapaths']['cost_benefits_calculator'][self.scenario_data['cost_benefits_calculator']]['cost_benefits_calculator_module'])
        self.cost_benefit_calculation_data = dataio.load_dataframe_from_csv(self.config['scenario_datapaths']['res_subsidies'][self.scenario_data['res_subsidies']]['cost_benefit_calculations_data_path'])
        # create the market object to work with
        self.market = electricity_market.ElectricityMarket(market_init_data)
        # handle any other config data as needed
        self.last_state = None
        self.current_state = None
        self.last_result = False
        self.profits=[]
        # the savepath/savename can be overriden
        if result_save_path: self.result_save_path = result_save_path
        else: self.result_save_path = self.config['system_datapaths']['result_save_path']
        self.charts_save_path = self.config['system_datapaths']['charts_save_path']
        self.save_results = self.config['program_options']['save_results']
        self.temp_save_interval = self.config['program_options']['temp_save_interval']
        self.temp_file = self.config['program_options']['temp_file_path']
        # this is used to save the actual start & end time of modelling for performance reasons etc.
        self.actual_time_elapsed = {'start':datetime.datetime.now(),'end':datetime.datetime.now()}
        # this is used to save the period currently modelled
        self.modelling_period = {'start':datetime.datetime.now(),'end':datetime.datetime.now()}
        # these attributes are to record the temp state of the modelling
        # to be used in recovering from a crash
        self.model_start_date = False
        self.model_days_to_run = False
        self.last_modelled_day_index = False
        self.last_modelled_day = False
        # do we zip saved files?
        self.zip_saves = self.config['program_options']['zip_saves']
        # are we using a temp file?
        self.use_temp = bool(int(self.config['program_options']['use_temp']))
        # if we should load the latest temp files and keep simming after a failure, do so right now
        if bool(int(self.config['program_options']['load_temp'])):
            # first load old main obj attributes and replace the current ones with them
            self.__dict__.update(dataio.load_object(self.temp_file, zipped=self.zip_saves))
            # then resume the modelling!!
            print ('Resuming modelling from %s-%s-%s' %(self.last_modelled_day.day,self.last_modelled_day.month,self.last_modelled_day.year))
            # calculate the remaining days to run
            remaining_days = self.model_days_to_run - self.last_modelled_day_index
            print ('Remaining %s days to model' %(remaining_days))
            # and start modelling from this point on!
            self.model_period(self.last_modelled_day+datetime.timedelta(days=1),self.modelling_period['end'],loaded_temp=True,start_day=self.modelling_period['start'])

        # this disables randomness additionally - to be used for verification purposes (via simpler milp model) so that two runs can be identical
        if self.config['program_options']['verification_mode']:
            # start by enabling everything
            self.market.generator_long_term_availability.loc[:] = 1
            for agent in self.market.single_plant_agents:
                # set hc = cc for all units
                agent.plant.min_uptime = 0
                agent.plant.min_downtime = 0
                agent.plant.start_costs[1] = agent.plant.start_costs[0]
                # and tcold = 0
                agent.plant.tcold = 0
                # disable imports & hydro via availability
                if agent.plant.kind in ['hydro','virtual','imports']:
                    self.market.generator_long_term_availability.loc[:,agent.plant.name] = 0
                    agent.plant.available_power[:] = agent.plant.pmax * 0
                    agent.plant.available_min_power[:] = agent.plant.pmin * 0
                    agent.plant.update_available_pmax_pmin()
            # also set all unit availabilities to binary 0 or 1 (use 1 if >= 0.5)
            self.market.generator_long_term_availability = (self.market.generator_long_term_availability + 0.001).round()


    def model_day(self,day):
        """
        Models one day, the day must be provided in datetime format.
        Also it must know if it is the last day to model (last_day=True)
        Calls the self.market.calculate_unit_commit function to generate data and updates
        the market and market state as needed
        """
        print ('=================================')
        print (day.strftime('Starting calculations for %d/%m/%Y'))
        # at init day, the agent has an action chosen already. no need to find it
        # besides, with no reference, it would be meaningless
        # at any other day get a new action for each agent according to his profits
        print ('Choosing new actions for all agents')
        # also report if updating policies via lspi for the first agent
        if self.market.single_plant_agents[0].learning_algorithm == 'lspi':
            if (len(self.market.single_plant_agents[0].learner.unused_samples)+1)%self.market.single_plant_agents[0].learner.lspi_update_frequency == 0:
                print("Updating LSPI policies")
        if type(self.last_result) is not bool:
            for agent in self.market.single_plant_agents:
                # calculate the profit of the last action
                profit = agent.calculate_profit(self.last_result[2])
                # update the taken_actions list of the agent.
                agent.taken_actions=agent.taken_actions.append(pandas.DataFrame([[agent.learner.current_action[0],agent.learner.current_action[1],profit]],index=[day-datetime.timedelta(days=1)],columns=['action','exploit_status','profit']))
                # choose a new action (plant data automatically updated)
                agent.choose_new_action(profit, self.last_state, self.current_state)
                # reset any profit penalties so that next day we'll start clean
                agent.profit_penalty = 0

        # run the uc model and get the result of the market clearance
        # before doing that, get a copy of the plants in order to restore & retry if uc does not find a solution
        backup_agents = copy.deepcopy(self.market.single_plant_agents)
        result = self.market.calculate_unit_commit(day,self.modelling_period['start'])

        # if an error was encountered, retry twice more before reporting it and stopping
        if type(result) == str:
            if result == 'final_dispatch_calculation_failure':
                print ('Unitcommit Error: Retrying...')
                # restore the agents & their plants & retry
                self.market.single_plant_agents = backup_agents
                result = self.market.calculate_unit_commit(day,self.modelling_period['start'])
        if type(result) == str:
                self.error(result)

        # save last result
        self.last_result = result
        # update the fuel prices history with what happened
        self.market.update_fuel_history(self.last_result,day)
        # now get the current state and update the last state
        self.last_state = self.current_state
        self.current_state = self.market.get_state(day)
        # if it is the last day, we need to know
        last_day = (day == self.modelling_period['end'])
        # if the month changes, apply the monthly updates needed
        if day.month != (day + datetime.timedelta(days=1)).month:
            self.market.do_monthly_updates(day,self.modelling_period['start'],last_day)
        # if the year changes, also apply yearly updates
        if day.year != (day + datetime.timedelta(days=1)).year:
            self.market.do_yearly_updates(day,last_day)
        # if this is the last day, run anything further needed to run
        if last_day:
            self.market.do_last_day_updates(day,self.last_result[2])

    def model_period(self,current_date,end_date,exit_when_done=True,loaded_temp=False,start_day=False):
        """
        Models the period specified, current_date and end_date must be provided in datetime format.
        It splits the period to model into days and models them via the self.model_day function.
        This function saves the results & lspi policies if needed, at the end of the process.
        """
        # convert date to datetime if needed
        if type(current_date) is not datetime.datetime:
            current_date = datetime.datetime.strptime(current_date,'%d/%m/%Y')
        if type(end_date) is not datetime.datetime:
            end_date = datetime.datetime.strptime(end_date,'%d/%m/%Y')

        days_to_run = (end_date - current_date).days + 1
        # error checking
        if end_date > self.market.demand.day.iloc[-1].to_pydatetime():
            self.error('There is no demand data about the whole day range')
        # inform about action
        print('Starting modelling, from %s-%s-%s to %s-%s-%s' %(current_date.day,current_date.month,current_date.year,end_date.day,end_date.month,end_date.year))
        self.actual_time_elapsed['start'] = datetime.datetime.now()
        print ('Start time is: %s' %self.actual_time_elapsed['start'])
        # run model for the days needed
        if current_date.year not in self.config['program_options']['valid_years_to_model']:
            print ('WARNING. The initial day is not within the allowed period to get modelled!!')
            print ('WMSIM will try to initialize on this day and will skip any other days within the non-allowed period!!\n')
        # do initial updates & run the initial day - to initialize the agents
        if not loaded_temp:
            self.modelling_period = {'start':current_date,'end':end_date}
            self.market.initialize_market(current_date,self.modelling_period['start'],last_day=False)
        self.model_day(current_date)
        # now run the rest of the days
        for day in range (days_to_run-1):
            day_to_model = current_date + datetime.timedelta(days=day+1)
            # skip any days within years not modelled
            if day_to_model.year in self.config['program_options']['valid_years_to_model']:
                self.model_day(day_to_model)
            else:
                print ('Skipping %s as it is outside the modelled boundary set' %day_to_model.date())
                # at the last day of the year, do updates
                if (day_to_model.month == 12 and day_to_model.day == 31):
                    if not (day_to_model == end_date) and self.market.total_hydro_capacity > 0:
                        self.market.fuel.randomize_reservoir_levels()
                        self.market.fuel.update_water_price(day_to_model+datetime.timedelta(days=1),self.market.single_plant_agents)

            # if we are at the save interval, save the temp data so that we can resume with minimal trouble
            if bool(int(self.use_temp)):
                if day_to_model.year in self.config['program_options']['valid_years_to_model']:
                    if day%self.temp_save_interval == 0:
                        print ('Saving tempfile')
                        self.model_start_date = current_date
                        self.model_days_to_run = days_to_run
                        self.last_modelled_day_index = day + 2
                        self.last_modelled_day = day_to_model
                        dataio.save_object(self.__dict__,self.config['program_options']['temp_file_path'],zipped=self.zip_saves)
                    print ('New tempfile save after', self.temp_save_interval-day%self.temp_save_interval, 'days')
        # if we need to save the learner policy, it is time to do so
        if self.config['program_options']['save_learner_policies']:
                print('Saving policies')
                dataio.save_all_agents_policies(self.config['program_options']['learning_algorithm'], self.market.single_plant_agents)
        # print performance msg
        print ('Calculations finished')
        self.actual_time_elapsed['end'] = datetime.datetime.now()
        print ('End time is: %s' % self.actual_time_elapsed['end'])
        print ('Elapsed time: %s' % (self.actual_time_elapsed['end'] - self.actual_time_elapsed['start']))
        print ('Time taken per day:',(self.actual_time_elapsed['end'] - self.actual_time_elapsed['start'])/days_to_run)

        # results contain the modeled period - everything is saved into the market object.
        results = self.market
        # save the results if required
        if self.save_results:
            result_save_path = self.result_save_path
            if self.zip_saves:
                result_save_path += '.'+self.zip_saves
            print ('Saving results in %s' %(result_save_path))
            dataio.save_object(results,self.result_save_path,zipped=self.zip_saves)
        # and clear the temp file if it was created
        if bool(int(self.use_temp)):
            temp_filepath = self.config['program_options']['temp_file_path']
            dataio.delete_file(temp_filepath,self.zip_saves)

        # this is an optional switch to make the program exit after calculations to avoid unwanted behaviour
        # where a second modelling could be required - which would result to both modelling results being aggregated
        if exit_when_done:
            exit()
        return results

    def error(self, errormsg):
        """
        Handle error reporting by printing the errormsg. Then quit.
        """
        print (errormsg)
        exit()

    def choose_scenario_files(self):
        """
        Depending on the scenario codename passed, choose the appropriate datapath files
        Scenario must be described by str codes where each section is separated by , while the value from its subsection by -
        """
        # load and unpack the scenario
        scenario = self.config['program_options']['scenario']
        for element in scenario:
            data = element.split('-')
            self.scenario_data.update({data[0]:data[1]})

        # now create a dict and add to it all the paths needed to create the market and run the program
        market_init_data = {}
        # those are standard (no scenario)
        market_init_data.update({'use_multiprocessing':self.config['program_options']['use_multiprocessing']})
        market_init_data.update({'enabled_generator_kinds':self.config['program_options']['enabled_generator_kinds']})
        market_init_data.update({'use_res_exports_plant':self.config['program_options']['use_res_exports_plant']})
        market_init_data.update({'valid_years_to_model':self.config['program_options']['valid_years_to_model']})
        market_init_data.update({'uc_verbosity':self.config['program_options']['uc_verbosity']})
        market_init_data.update({'learning_algorithm':self.config['program_options']['learning_algorithm']})
        market_init_data.update({'uc_module':self.config['program_options']['uc_module']})
        market_init_data.update({'load_learner_policies':self.config['program_options']['load_learner_policies']})
        market_init_data.update({'agent_actions_data_path':self.config['agents_module_datapaths']['agent_actions_data_path']})
        market_init_data.update({'agent_data_data_path':self.config['agents_module_datapaths']['agent_data_data_path']})
        market_init_data.update({'lspi_policy_path':self.config['agents_module_datapaths']['lspi_policy_path']})
        market_init_data.update({'generators_path':self.config['scenario_datapaths']['generators_path']})
        market_init_data.update({'market_data_path':self.config['scenario_datapaths']['market_data_path']})
        market_init_data.update({'reserves_requirements_path':self.config['scenario_datapaths']['reserves_requirements_path']})
        market_init_data.update({'smp_data_path':self.config['scenario_datapaths']['smp_data_path']})
        market_init_data.update({'import_plants_data_path':self.config['scenario_datapaths']['import_plants_data_path']})
        market_init_data.update({'nat_gas_contribution_path':self.config['scenario_datapaths']['nat_gas_contribution_path']})
        # and those are part of the scenario
        market_init_data.update({'generator_availability_path':self.config['scenario_datapaths']['generator_availability'][self.scenario_data['generator_availability']]['generator_data_path']})
        market_init_data.update({'generators_biomass_cofiring_path':self.config['scenario_datapaths']['generators_biomass_cofiring'][self.scenario_data['generators_biomass_cofiring']]['biomass_cofiring_path']})
        market_init_data.update({'demand_path':self.config['scenario_datapaths']['demand'][self.scenario_data['demand']]['demand_data_path']})
        market_init_data.update({'solar_scenario_path':self.config['scenario_datapaths']['solar_generation'][self.scenario_data['solar_generation']]['scenario_generation_path']})
        market_init_data.update({'solar_park_percentages_path':self.config['scenario_datapaths']['solar_park_percentages'][self.scenario_data['solar_park_percentages']]['solar_park_percentages_path']})
        market_init_data.update({'solar_growth_path':self.config['scenario_datapaths']['solar_generation'][self.scenario_data['solar_generation']]['scenario_growth_path']})
        market_init_data.update({'wind_onshore_scenario_path':self.config['scenario_datapaths']['wind_onshore_generation'][self.scenario_data['wind_onshore_generation']]['scenario_generation_path']})
        market_init_data.update({'wind_onshore_growth_path':self.config['scenario_datapaths']['wind_onshore_generation'][self.scenario_data['wind_onshore_generation']]['scenario_growth_path']})
        market_init_data.update({'wind_offshore_scenario_path':self.config['scenario_datapaths']['wind_offshore_generation'][self.scenario_data['wind_offshore_generation']]['scenario_generation_path']})
        market_init_data.update({'wind_offshore_growth_path':self.config['scenario_datapaths']['wind_offshore_generation'][self.scenario_data['wind_offshore_generation']]['scenario_growth_path']})
        market_init_data.update({'hydro_generation_path':self.config['scenario_datapaths']['hydro_generation'][self.scenario_data['hydro_generation']]['hydro_generation_path']})
        market_init_data.update({'hydro_plant_max_generation_path':self.config['scenario_datapaths']['hydro_generation'][self.scenario_data['hydro_generation']]['hydro_plant_max_generation_path']})
        market_init_data.update({'hydro_basins_levels_path':self.config['scenario_datapaths']['hydro_generation'][self.scenario_data['hydro_generation']]['hydro_basins_levels_path']})
        market_init_data.update({'hydro_basins_level_randomization':self.config['scenario_options']['hydro_basins_level_randomization'][self.scenario_data['hydro_basins_level_randomization']]['hydro_basins_level_randomization']})
        market_init_data.update({'hydro_basins_path':self.config['scenario_datapaths']['hydro_generation'][self.scenario_data['hydro_generation']]['hydro_basins_path']})
        market_init_data.update({'hydro_price_variables_path':self.config['scenario_datapaths']['hydro_generation'][self.scenario_data['hydro_generation']]['hydro_price_variables_path']})
        market_init_data.update({'thermal_reference_price_path':self.config['scenario_datapaths']['hydro_generation'][self.scenario_data['hydro_generation']]['thermal_reference_price_path']})
        market_init_data.update({'import_prices_path':self.config['scenario_datapaths']['import_prices'][self.scenario_data['import_prices']]})
        market_init_data.update({'carbon_price_path':self.config['scenario_datapaths']['carbon_price'][self.scenario_data['carbon_price']]['carbon_price_path']})
        market_init_data.update({'import_prices_randomization':self.config['scenario_options']['import_prices_randomization'][self.scenario_data['import_prices_randomization']]['import_prices_randomization']})
        market_init_data.update({'import_availabilities_path':self.config['scenario_datapaths']['imports_availability'][self.scenario_data['imports_availability']]['imports_availability_path']})
        market_init_data.update({'res_prices_yearly_data_path':self.config['scenario_datapaths']['res_prices_yearly_path'][self.scenario_data['res_prices_yearly_path']]['res_pricing_path']})
        market_init_data.update({'nat_gas_scenario_path':self.config['scenario_datapaths']['natural_gas_prices_path'][self.scenario_data['natural_gas_prices_path']]['natural_gas_prices_path']})
        return market_init_data

    def visualize_results(self, results_path=None, \
                                save_folder='', \
                                save_results=True, \
                                data_temporal_resolution = 'M', \
                                show_plots=True,\
                                market=None, \
                                agent_results=True, \
                                agent_plots = ['actions','actions_per_plant_kind','profit_per_action','profit_per_plant_kind'], \
                                market_results = True, \
                                market_plots_regression = False, \
                                tables = True, \
                                table_data_temporal_resolution = 'M', \
                                market_plots = ['demand_and_electricity_mix',\
                                        'electricity_mix', \
                                        'electricity_mix_percentages',\
                                        'imports_exports',\
                                        'res_production',\
                                        'virtual_production',\
                                        'prices'], \
                                calculate_tax_benefits = False):
        """
        loads up the market object from the specified path or uses a given market object
        and calls the related data extractor funtions to plot agent and/or market results
        For agents, if the mode is 'total', results are aggregated across all agents and the whole timeframe.
        If the mode is 'daily', results are aggregated across all agents, but for each day separately.

        For markets, available plots are: 'electricity_mix','prices','smp_vs_demand','res_production','thermal_vs_water_vs_res','everything',
        and they must be provided as a list (with 1 or more elements).
        Also it is possible to specify custom plots, by specifying selected columns from the
        electricity market results dataframe, specified as a list of stringnames (e.g. ['day','thermal','res','water_power'])
        """
        # load the market obj if needed
        if market == None:
            print('Loading market object for visualizations')
            market = data_extractor.load_market(results_path, self.zip_saves)
        print('Getting results into tables')
        # save the generated results in this dict for reuse within this funct
        results = {'agent_results_data':False,'market_results_data':False}
        # get agent results
        agent_results_data = data_extractor.get_agent_results(market)
        results.update({'agent_results_data':agent_results_data})
        # get market results
        market_results_data = data_extractor.get_market_results(market,cost_benefits_calculator,self.cost_benefit_calculation_data)
        results.update({'market_results_data':market_results_data})

        # get actual system price
        market_tables_annual = data_extractor.generate_market_tables(market,results,save_results=False,save_path=self.charts_save_path,save_folder=save_folder,data_temporal_resolution='A')
        market_tables_hourly = data_extractor.generate_market_tables(market,results,save_results=False,save_path=self.charts_save_path,save_folder=save_folder,data_temporal_resolution='H')
        system_prices = cost_benefits_calculator.calculate_actual_system_prices(market,results['market_results_data'],market_tables_hourly,self.cost_benefit_calculation_data,save_results=save_results,save_path=self.charts_save_path,save_folder=save_folder)
        # update results with the actual system price
        market_results_data.loc[:,'actual_system_price'] = system_prices.values

        print('Plotting, post-processing, and saving')
        # now use this for plotting/data extraction
        # start with tables
        if tables:
            market_tables = data_extractor.generate_market_tables(market,results,save_results=save_results,save_path=self.charts_save_path,save_folder=save_folder,data_temporal_resolution=table_data_temporal_resolution)
        # plot agent results if required
        if agent_results:
            data_extractor.plot_agent_results(results['agent_results_data'],show_plots=show_plots,save_folder=save_folder,save_path=self.charts_save_path,figures=agent_plots,save_results=save_results)
        # plot market results if required
        if market_results:
            data_extractor.plot_market_results(results['market_results_data'],market_plots,show_plots=show_plots,save_results=save_results,save_path=self.charts_save_path,save_folder=save_folder,data_temporal_resolution=data_temporal_resolution,market_plots_regression=market_plots_regression)
        if calculate_tax_benefits:
            tax_cost_benefits = cost_benefits_calculator.calculate_tax_benefits(market,market_tables_annual,results,self.cost_benefit_calculation_data,save_results=save_results,save_path=self.charts_save_path,save_folder=save_folder)


# RUN THE MODEL
# for initial run
override_program_options0 = {'save_results':True,
                            'learning_algorithm':'lspi',
                            'uc_module':'epl',
                            'save_learner_policies':True,
                            'load_learner_policies':False,
                            'clear_learner_policies':True,
                            'use_temp':False,
                            'load_temp':False,
                            'uc_verbosity':0,
                            'verification_mode':0
                            }
# for using saved policies without updating them - real runs
override_program_options1 = {'save_results':True,
                            'learning_algorithm':'lspi',
                            'uc_module':'epl',
                            'save_learner_policies':False,
                            'load_learner_policies':True,
                            'clear_learner_policies':False,
                            'use_temp':True,
                            'load_temp':False,
                            'uc_verbosity':0,
                            'verification_mode':0
                            }

# for using temp loading - real runs
override_program_options2 = {'save_results':True,
                            'learning_algorithm':'lspi',
                            'uc_module':'epl',
                            'save_learner_policies':False,
                            'load_learner_policies':True,
                            'clear_learner_policies':False,
                            'use_temp':True,
                            'load_temp':True,
                            'uc_verbosity':0,
                            'verification_mode':0
                            }

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # do a run for results
    main = Main(init_file='data/greece_baseline_ipto.ini',result_save_path='data/greece/results/renewable_penetration_2022_2030.pkl', override_program_options=override_program_options1)
    main.model_period('1/1/2022','31/1/2022',exit_when_done=False)
    main.visualize_results(results_path='data/greece/results/renewable_penetration_2022_2030.pkl',save_folder='renewable_penetration_2022_2030',show_plots=True,agent_results=True,market_results=True,tables=True,data_temporal_resolution='A',table_data_temporal_resolution ='A')
