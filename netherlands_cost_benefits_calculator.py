"""
This module is used to conduct cost-benefit calculations for the dutch electricity market based on current subsidies framework & res projections
"""

import numpy
import pandas
import dataio
import datetime

def calculate_tax_benefits(market,market_tables,results,cost_benefit_calculation_data,save_results,save_path,save_folder):
    """
    This function calculates the tax benefits or costs across the 3 separate aspects and saves the results as needed
    The aspects consider:a. Taxation of electricity producers & imports for produced power
                         b. Taxation of self-consumed electricity (lost profit)
                         c. Taxation on the whole RES sector
    """
    # get the electricity_producers_tax_data
    electricity_producers_tax_data = calculate_electricity_producers_tax(market,market_tables,cost_benefit_calculation_data)
    # unpack them for reuse later on
    intermittent_generation,taxes,taxes_concat,subsidies,subsidies_concat = electricity_producers_tax_data
    # get the tax loss due to self-consumtion
    electricity_consumers_tax_loss = calculate_electricity_consumers_tax(market_tables[0].loc[:,'calculated_smp'].dropna(),intermittent_generation.loc[:,'pv_power_rooftop'],cost_benefit_calculation_data)
    # and get the sector tax
    res_sector_tax = calculate_res_sector_tax(market,market_tables,cost_benefit_calculation_data,intermittent_generation)
    # also sum up into a new df
    totals = pandas.DataFrame(index=taxes_concat.index)
    totals.loc[:,'tax_income'] = taxes_concat.sum(axis=1)
    totals.loc[:,'subsidies_cost'] = subsidies_concat.sum(axis=1)
    totals.loc[:,'tax_loss_due_to_self_consumtion'] = electricity_consumers_tax_loss
    if save_results:
            # create the folder to save if needed
            root_folder_path = save_path+save_folder
            dataio.create_folder(root_folder_path)
            # and save the data in an excel multisheet
            writer = pandas.ExcelWriter(root_folder_path+'/tax_costs_benefits.xlsx')
            taxes_concat.to_excel(writer,'tax_income')
            subsidies_concat.to_excel(writer,'subsidies_cost')
            intermittent_generation.to_excel(writer,'intermittent_generation')
            totals.to_excel(writer,'sums')
            writer.save()
    return [intermittent_generation,taxes,taxes_concat,subsidies,subsidies_concat,electricity_consumers_tax_loss]

def calculate_electricity_producers_tax(market,market_tables,cost_benefit_calculation_data):
    """
    Calculate the tax on electricity producers. Conventionals, RES & res_thermal all are handled differently
    First everyone's profitability is calculated/loaded, then the tax system is simulated
    """
    # unpack market_tables
    market_results_resampled_mean,market_results_resampled_aggregated,agent_results_resampled_mean,agent_results_resampled_aggregated = market_tables
    # get generation for subsidized producers & unpack
    plant_profitability,intermittent_res_generation,subsidized_thermal_power,subsidized_cofiring_power,subsidized_solar_parks_power,subsidized_w_onshore_power,subsidized_w_offshore_power = calculate_subsidized_electricity_producers_generation(market,market_tables,cost_benefit_calculation_data)
    # go on to get the profitabilities for conventional plants
    # separate by type, since these are taxed separately, loading and calculating profitability
    fossil_plant_types = ['lignite-st','nat-gas-st','hard_coal','nat-gas-ccgt','nat-gas-occgt','nuclear']
    plant_profitability_fossil = plant_profitability.loc(axis=1)[:,fossil_plant_types,:]
    plant_profitability_imports = plant_profitability.loc(axis=1)[:,'imports',:]

    # Profitability is assumed to equal the non-subsidized part, but only after the subsidy has expired
    # thus we track the % of installed capacity eligible & ineligible for subsidy
    # the eligible plants will incur a treasury loss equal to the subsidy
    # and the ineligible ones will incur a tax gain equal to 20-25% of the profit
    # it is assumed that new capacities at scenario start (2015) were just installed thus eligible for subsidy
    # thermal plants are those types mentioned in the lists below
    # and calculate the tax gains for everything
    fossil_taxes = calculate_tax_from_profitability(plant_profitability_fossil,cost_benefit_calculation_data,'generation_companies')
    import_taxes = calculate_tax_from_profitability(plant_profitability_imports,cost_benefit_calculation_data,'imports')

    thermal_cost,thermal_taxes = calculate_subsidized_power_tax_and_cost(subsidized_thermal_power,cost_benefit_calculation_data,'thermal')
    biomass_cofiring_cost,biomass_cofiring_taxes = calculate_subsidized_power_tax_and_cost(subsidized_cofiring_power,cost_benefit_calculation_data,'biomass_cofiring')
    solar_parks_cost,solar_parks_taxes = calculate_subsidized_power_tax_and_cost(subsidized_solar_parks_power,cost_benefit_calculation_data,'solar')
    wind_onshore_cost,wind_onshore_taxes = calculate_subsidized_power_tax_and_cost(subsidized_w_onshore_power,cost_benefit_calculation_data,'wind_onshore')
    wind_offshore_cost,wind_offshore_taxes = calculate_subsidized_power_tax_and_cost(subsidized_w_offshore_power,cost_benefit_calculation_data,'wind_offshore')

    # fix indexes, pack and return
    fossil_taxes.columns = fossil_taxes.columns.get_level_values(0)
    fossil_taxes.index = [date.year for date in fossil_taxes.index]
    import_taxes.columns = import_taxes.columns.get_level_values(0)
    import_taxes.index = [date.year for date in import_taxes.index]
    taxes = [fossil_taxes,import_taxes,thermal_taxes,biomass_cofiring_taxes,solar_parks_taxes,wind_onshore_taxes,wind_offshore_taxes]
    taxes_concat = pandas.concat(taxes,axis=1)
    subsidies = [thermal_cost,biomass_cofiring_cost,solar_parks_cost,wind_onshore_cost,wind_offshore_cost]
    subsidies_concat = pandas.concat(subsidies,axis=1)
    results = [intermittent_res_generation,taxes,taxes_concat,subsidies,subsidies_concat]
    return results

def calculate_electricity_consumers_tax(yearly_smp,pv_power_rooftop,cost_benefit_calculation_data):
    """
    Calculate taxation lost due to consumers self-producing net-metered power
    This is the taxes part on top of the smp mutiplied by the self-produced power
    """
    # this tax rate is the part of the final price that is not the smp. thus the smp is (1-electricity_tax_rate) *100 %
    electricity_tax_rate = cost_benefit_calculation_data.loc['electricity_tax_rate','value']
    taxes_part_of_price = (yearly_smp / (1-electricity_tax_rate)) * electricity_tax_rate
    taxes_part_of_price.index = [date.year for date in taxes_part_of_price.index]
    pv_taxes_loss = pv_power_rooftop * taxes_part_of_price
    return pv_taxes_loss

def calculate_res_sector_tax(market,market_tables,cost_benefit_calculation_data,intermittent_generation):
    """
    This is used to calculate the taxes gained from the whole (booming presumably) res sector
    To do so, we will do a simple assumption for the profitability of companies as a percent of the final value, and tax that
    And we will find the final value for each year using an experience curve
    """
    profitability = pandas.DataFrame(index=market.fuel.pv_growth.index,columns=['profits_solar','profits_wind_onshore','profits_wind_offshore'])
    # load the initial installed capacity
    initial_solar_installed_capacity = cost_benefit_calculation_data.loc['solar_res_initial_installed_capacity_mw','value']
    solar_lr = cost_benefit_calculation_data.loc['solar_res_installation_cost_learning_rate','value']

    initial_wind_onshore_installed_capacity = cost_benefit_calculation_data.loc['wind_onshore_res_initial_installed_capacity_mw','value']
    wind_onshore_lr = cost_benefit_calculation_data.loc['wind_onshore_res_installation_cost_learning_rate','value']

    initial_wind_offshore_installed_capacity = cost_benefit_calculation_data.loc['wind_onshore_res_initial_installed_capacity_mw','value']
    wind_offshore_lr = cost_benefit_calculation_data.loc['wind_offshore_res_installation_cost_learning_rate','value']

    for year in market.fuel.pv_growth.index:
        yearly_solar_installed_capacity = initial_solar_installed_capacity * market.fuel.pv_growth.loc[year,'growth']

        solar_b = -numpy.log(1-solar_lr)/numpy.log(2)
        final_solar_cost = cost_benefit_calculation_data.loc['solar_res_initial_installation_cost_mw','value'] * pow((yearly_solar_installed_capacity/initial_solar_installed_capacity),solar_b)
        final_solar_profitability = final_solar_cost * cost_benefit_calculation_data.loc['assumed_energy_res_supply_chain_profitability_percent_of_final_value','value']
        profitability.loc[year,'profits_solar'] = final_solar_profitability.mean()

        yearly_wind_onshore_installed_capacity = initial_wind_onshore_installed_capacity * market.fuel.wind_onshore_growth.loc[year,'growth']
        wind_onshore_b = -numpy.log(1-wind_onshore_lr)/numpy.log(2)
        final_wind_onshore_cost = cost_benefit_calculation_data.loc['wind_onshore_res_initial_installation_cost_mw','value'] * pow((yearly_wind_onshore_installed_capacity/initial_wind_onshore_installed_capacity),wind_onshore_b)
        final_wind_onshore_profitability = final_wind_onshore_cost * cost_benefit_calculation_data.loc['assumed_energy_res_supply_chain_profitability_percent_of_final_value','value']
        profitability.loc[year,'profits_wind_onshore'] = final_wind_onshore_profitability.mean()

        yearly_wind_offshore_installed_capacity = initial_wind_offshore_installed_capacity * market.fuel.wind_offshore_growth.loc[year,'growth']
        wind_offshore_b = -numpy.log(1-wind_offshore_lr)/numpy.log(2)
        final_wind_offshore_cost = cost_benefit_calculation_data.loc['wind_offshore_res_initial_installation_cost_mw','value'] * pow((yearly_wind_offshore_installed_capacity/initial_wind_offshore_installed_capacity),wind_offshore_b)
        final_wind_offshore_profitability = final_wind_offshore_cost * cost_benefit_calculation_data.loc['assumed_energy_res_supply_chain_profitability_percent_of_final_value','value']
        profitability.loc[year,'profits_wind_offshore'] = final_wind_offshore_profitability.mean()
    sector_taxes = calculate_tax_from_profitability(profitability,cost_benefit_calculation_data,'generation_companies')
    return sector_taxes


def calculate_subsidized_electricity_producers_generation(market,market_tables,cost_benefit_calculation_data):
    """
    Calculate generation of subsidized res. Those have a subsidized & non-subsidized part and both are needed
    """
    # unpack market_tables
    market_results_resampled_mean,market_results_resampled_aggregated,agent_results_resampled_mean,agent_results_resampled_aggregated = market_tables

    # for conventional plants (non subsidized), the profitability has already been recorded in agent_results_resampled_aggregated, so extract it
    # we need to adjust it, as biomass cofiring alters its values
    plant_profitability = agent_results_resampled_aggregated.loc(axis=1)[:,:,'profit'].copy()
    # co-firing plants, are subsidized, so we need to find out the co-firing energy part and reduce the profitability by the co-firing part * smp
    # still we need to go over the sums again to reduce properly the profits and calculate the subsidies
    cofiring_generation = pandas.DataFrame()
    for agent in market.single_plant_agents:
        generator_name = agent.plant.name
        if generator_name in market.generators_biomass_cofiring.columns and market.generators_biomass_cofiring.loc[:,generator_name].sum() > 0:
            # get the hourly generation & profit
            generation = agent.plant.saved_online_data.loc[:,'power'].drop(24,level=1)
            generation.index = generation.index.get_level_values(0)
            plant_profit = generation * market.fuel.calculated_smp.hourly_smp.values
            # convert to yearly sums
            index = numpy.arange(generation.index[0].year,generation.index[-1].year+1)
            generation = generation.resample('A').sum()
            plant_profit = plant_profit.resample('A').sum()
            generation.index = index
            plant_profit.index = index
            # multiply with the percent to get correct amounts of generation & profit (adjust for allowed generation)
            cofiring_percents = market.generators_biomass_cofiring.loc[:,generator_name].values
            plant_profit = plant_profit * cofiring_percents[:len(plant_profit)]
            generation = generation * cofiring_percents[:len(plant_profit)]
            # add the col to cofiring_generation
            cofiring_generation.loc[:,generator_name] = generation
            # reduce the conventional profits by the co-firing amount.
            generator_multiindex = plant_profitability.loc(axis=1)[generator_name,:,'profit'].columns[0]
            plant_profitability.loc[:,generator_multiindex] -= plant_profit.values

    thermal_res = ['waste','biomass','chp','biomass_cofiring']
    # first get the availabilities
    thermal_installed_capacities_names = agent_results_resampled_aggregated.loc(axis=1)[:,thermal_res,'power'].columns.get_level_values(0)
    thermal_installed_capacities = market.generator_long_term_availability.loc[:,thermal_installed_capacities_names]
    thermal_installed_capacities = thermal_installed_capacities.loc[thermal_installed_capacities.index.isin(market.valid_years_to_model)]
    # then find out the subsidized & unsubsidized capacities as a % of total
    subsidized_capacities_percents_thermal = calculate_subsidized_percentages(market.valid_years_to_model,cost_benefit_calculation_data.loc['thermal_res_subsidy_start','value'],cost_benefit_calculation_data.loc['thermal_res_subsidy_duration','value'],cost_benefit_calculation_data.loc['all_subsidies_expiry','value'],thermal_installed_capacities)

    # get the generation totals
    plant_generation_thermal_res = market_results_resampled_aggregated.loc[:,market_results_resampled_aggregated.columns.isin(thermal_res)].sum(axis=1).copy()
    plant_generation_thermal_res.index = [date.year for date in plant_generation_thermal_res.index]
    # and multiply with subsidized_capacities_percents to get the subsidized & unsubsidized generation part
    subsidized_thermal_power = subsidized_capacities_percents_thermal.multiply(plant_generation_thermal_res,axis=0).fillna(0)

    # do the same for co-firing
    # start by aggregating all plants generation
    cofiring_generation = cofiring_generation.sum(axis=1)
    # if a limit exists, adjust the percents correctly using a cofiring_percent_adjustment percentile variable to not allow more cofiring than described
    if cost_benefit_calculation_data.loc['biomass_cofiring_res_max_yearly_generation','value'] >= 0:
        cofiring_percent_adjustment = (cost_benefit_calculation_data.loc['biomass_cofiring_res_max_yearly_generation','value']/cofiring_generation).clip(0,1)
        # apply the adjustments
        cofiring_generation = cofiring_generation*cofiring_percent_adjustment

    # if no cofiring takes place after subsidy expiry, adjust the generation further as needed
    if cost_benefit_calculation_data.loc['biomass_cofiring_res_max_yearly_generation_after_subsidy_expiry','value'] >= 0:
        # get the subsidized percentages for cofiring and reduce the generation if needed
        subsidized_capacities_percents_cofiring = calculate_subsidized_percentages(market.valid_years_to_model,cost_benefit_calculation_data.loc['biomass_cofiring_res_subsidy_start','value'],cost_benefit_calculation_data.loc['biomass_cofiring_res_subsidy_duration','value'],cost_benefit_calculation_data.loc['all_subsidies_expiry','value'],cofiring_generation.to_frame())
        cofiring_generation = cofiring_generation*subsidized_capacities_percents_cofiring.loc[:,'subsidized']

    # also, if subsidies are limited to a certain level of production, get the percentile value of that
    cofiring_subsidy_adjustment = pandas.Series(1,index=numpy.arange(cofiring_generation.index[0],cofiring_generation.index[-1]+1))
    if cost_benefit_calculation_data.loc['biomass_cofiring_res_max_yearly_generation_subsidies','value'] >= 0:
        cofiring_subsidy_adjustment = (cost_benefit_calculation_data.loc['biomass_cofiring_res_max_yearly_generation_subsidies','value']/cofiring_generation).clip(0,1)
    subsidized_capacities_percents_cofiring = calculate_subsidized_percentages(market.valid_years_to_model,cost_benefit_calculation_data.loc['biomass_cofiring_res_subsidy_start','value'],cost_benefit_calculation_data.loc['biomass_cofiring_res_subsidy_duration','value'],cost_benefit_calculation_data.loc['all_subsidies_expiry','value'],market.generators_biomass_cofiring)
    # aggregate removing duplicate indices
    subsidized_cofiring_power = subsidized_capacities_percents_cofiring.multiply(cofiring_generation,axis=0).fillna(0)

    # and do the same for the three intermittent res
    # get the generation summed. This is recalculated, so as to also take into account the power used in the exports plant
    pv_generation = pandas.DataFrame(market.fuel.pv_scenario.drop('datetime',axis=1).sum(axis=1))
    pv_generation.index = market.fuel.pv_scenario.loc[:,'datetime']
    wind_onshore_generation = pandas.DataFrame(market.fuel.wind_onshore_scenario.drop('datetime',axis=1).sum(axis=1))
    wind_onshore_generation.index = market.fuel.wind_onshore_scenario.loc[:,'datetime']
    wind_offshore_generation = pandas.DataFrame(market.fuel.wind_offshore_scenario.drop('datetime',axis=1).sum(axis=1))
    wind_offshore_generation.index = market.fuel.wind_offshore_scenario.loc[:,'datetime']

    intermittent_res_generation = pandas.DataFrame()
    for year in subsidized_thermal_power.index:
        if (str(year)+'-1-1') in market.fuel.calculated_smp.index:
            # pv power is split between rooftop and solar parks
            intermittent_res_generation.loc[year,'pv_power_rooftop'] = pv_generation.loc[str(year)+'-1-1':str(year)+'-12-31'].sum().sum() * market.fuel.solar_percents_per_type.loc[year,'solar_rooftop']
            intermittent_res_generation.loc[year,'pv_power_solar_parks'] = pv_generation.loc[str(year)+'-1-1':str(year)+'-12-31'].sum().sum() * market.fuel.solar_percents_per_type.loc[year,'solar_parks']
            intermittent_res_generation.loc[year,'wind_onshore'] = wind_onshore_generation.loc[str(year)+'-1-1':str(year)+'-12-31'].sum().sum()
            intermittent_res_generation.loc[year,'wind_offshore'] = wind_offshore_generation.loc[str(year)+'-1-1':str(year)+'-12-31'].sum().sum()

    # then get the subsidized & unsubsidized part for all three (rooftops are excluded due to being net-metered with 0 profit assumed)
    subsidized_capacities_percents_pv = calculate_subsidized_percentages(market.valid_years_to_model,cost_benefit_calculation_data.loc['solar_res_subsidy_start','value'],cost_benefit_calculation_data.loc['solar_res_subsidy_duration','value'],cost_benefit_calculation_data.loc['all_subsidies_expiry','value'],market.fuel.pv_growth)
    subsidized_solar_parks_power = subsidized_capacities_percents_pv.multiply(intermittent_res_generation.loc[:,'pv_power_solar_parks'],axis=0).fillna(0)

    subsidized_capacities_percents_w_onshore = calculate_subsidized_percentages(market.valid_years_to_model,cost_benefit_calculation_data.loc['wind_onshore_res_subsidy_start','value'],cost_benefit_calculation_data.loc['wind_onshore_res_subsidy_duration','value'],cost_benefit_calculation_data.loc['all_subsidies_expiry','value'],market.fuel.wind_onshore_growth)
    subsidized_w_onshore_power = subsidized_capacities_percents_w_onshore.multiply(intermittent_res_generation.loc[:,'wind_onshore'],axis=0).fillna(0)

    subsidized_capacities_percents_w_offshore = calculate_subsidized_percentages(market.valid_years_to_model,cost_benefit_calculation_data.loc['wind_offshore_res_subsidy_start','value'],cost_benefit_calculation_data.loc['wind_offshore_res_subsidy_duration','value'],cost_benefit_calculation_data.loc['all_subsidies_expiry','value'],market.fuel.wind_offshore_growth)
    subsidized_w_offshore_power = subsidized_capacities_percents_w_offshore.multiply(intermittent_res_generation.loc[:,'wind_offshore'],axis=0).fillna(0)

    # pack results and return
    results = [plant_profitability,intermittent_res_generation,subsidized_thermal_power,subsidized_cofiring_power,subsidized_solar_parks_power,subsidized_w_onshore_power,subsidized_w_offshore_power]
    return results

def calculate_subsidized_percentages(valid_years_to_model,subsidy_start_year,subsidy_duration,subsidy_expiry_year,installed_capacities,subsidy_adjustment=False):
    """
    For every valid year, find out the subsidized & unsubsidized capacities as a % of total
    Takes into account the expiry of subsidies for capacities after a time period,
    the general expiry of subsidies at some point,
    and the reduction of subsidies due to an imposed limit on subsidized generation
    """
    if type(subsidy_adjustment) is bool:
        subsidy_adjustment = 1
    # start by getting the unnormalized capacity diff & use it to fill subsidized & unsubsidized capacity cols
    subsidized_capacities_percents = pandas.DataFrame(0,index=valid_years_to_model,columns=['subsidized','unsubsidized'])
    for row in installed_capacities.iterrows():
        current_year = row[0]
        # if we are not past the subsidies expiry year, new subsidies are allowed
        if current_year <= subsidy_expiry_year:
            # before the first year, nothing going!
            if current_year < subsidy_start_year:
                subsidized_capacities_percents.loc[current_year,'subsidized'] = 0
                subsidized_capacities_percents.loc[current_year:,'unsubsidized'] = 1
            else:
                # in the first year, everything installed gets a subsidy
                if current_year == subsidy_start_year:
                    new_capacity = row[1].sum()
                else:
                    # in every other year, we only need to check if anything new has been installed
                    current_capacity = row[1].sum()
                    last_year = installed_capacities.index[installed_capacities.index.get_loc(current_year)-1]
                    last_capacity = installed_capacities.loc[last_year,:].sum()
                    new_capacity = max(0, (current_capacity - last_capacity))
                subsidized_capacities_percents.loc[current_year:current_year+subsidy_duration-1,'subsidized'] += new_capacity
                subsidized_capacities_percents.loc[current_year+subsidy_duration:,'unsubsidized'] += new_capacity

        # if not, we can set the rest of the percents as needed and break
        else:
            subsidized_capacities_percents.loc[current_year:installed_capacities.index[-1],'subsidized'] = 0
            subsidized_capacities_percents.loc[current_year:installed_capacities.index[-1],'unsubsidized'] = 1
            break

    # finally, we need to normalize the values across each row, so that they will sum to 1
    subsidized_capacities_percents = subsidized_capacities_percents.div(subsidized_capacities_percents.sum(axis=1),axis=0).dropna()
    # the subsidy values possibly need to be modified as per the subsidy adjustment, changing the subsidized & unsibsidized parts
    adjustment_values = subsidized_capacities_percents.loc[:,'subsidized'].multiply(1-subsidy_adjustment).fillna(0)
    subsidized_capacities_percents.loc[:,'subsidized'] -= adjustment_values
    subsidized_capacities_percents.loc[:,'unsubsidized'] += adjustment_values
    return subsidized_capacities_percents.round(5)

def calculate_tax_from_profitability(profitability,cost_benefit_calculation_data,tax_type):
    taxes = 0
    if tax_type in ['generation_companies']:
        low_tax_rate = cost_benefit_calculation_data.loc['generation_companies_tax_rate_low','value']
        high_tax_rate = cost_benefit_calculation_data.loc['generation_companies_tax_rate_high','value']
        zero_tax_profit_cutoff = cost_benefit_calculation_data.loc['generation_companies_tax_cutoff_low','value']
        low_tax_profit_cutoff = cost_benefit_calculation_data.loc['generation_companies_tax_cutoff_high','value']
        # now that we know the boundaries, add the taxes for each category
        taxes = profitability[profitability[profitability >= zero_tax_profit_cutoff] < low_tax_profit_cutoff].fillna(0) * low_tax_rate + \
                profitability[profitability > low_tax_profit_cutoff].fillna(0) * high_tax_rate
    elif tax_type in ['imports']:
        tax_rate = cost_benefit_calculation_data.loc['imports_tax_rate','value']
        # negative profits not taxed
        zero_tax_profit_cutoff = 0
        taxes = profitability[profitability > zero_tax_profit_cutoff].fillna(0) * tax_rate
    return taxes

def calculate_subsidized_power_tax_and_cost(generated_power,cost_benefit_calculation_data,tax_type):
    """
    Calculates both the cost of subsidies and the taxes gained for all subsidized energy types
    the generated_power is a dataframe with two parts: a subsidized and an unsubsidized one
    """
    subsidy_price = cost_benefit_calculation_data.loc[tax_type+'_res_subsidy_price_mwh','value']
    base_price = cost_benefit_calculation_data.loc[tax_type+'_res_base_price_mwh','value']
    # The cost equals the subsidized part * the subsidized part of the price
    cost = (generated_power.loc[:,'subsidized']*subsidy_price).to_frame(str(tax_type)+'_costs')
    # the taxed profitability is the unsubsidized part * the unsubsidized part of the price
    profitability_to_tax = generated_power.loc[:,'unsubsidized']*base_price
    # convert to dataframe for use in calculate_tax_from_profitability
    profitability_to_tax = profitability_to_tax.to_frame(str(tax_type)+'_taxes')
    # it is assumed that the taxation is the same for all companies
    taxes = calculate_tax_from_profitability(profitability_to_tax,cost_benefit_calculation_data,'generation_companies')
    return cost,taxes

def calculate_subsidized_power_total_cost(generated_power,cost_benefit_calculation_data,tax_type):
    subsidy_price = cost_benefit_calculation_data.loc[tax_type+'_res_subsidy_price_mwh','value']
    base_price = cost_benefit_calculation_data.loc[tax_type+'_res_base_price_mwh','value']
    unsubsidized_cost = (generated_power.loc[:,'unsubsidized']*base_price).to_frame(str(tax_type)+'_base_costs')
    subsidized_cost = (generated_power.loc[:,'subsidized']*(base_price+subsidy_price)).to_frame(str(tax_type)+'_subsidy_costs')
    return unsubsidized_cost,subsidized_cost

def calculate_actual_system_prices(market,market_results,market_tables,cost_benefit_calculation_data,save_results,save_path,save_folder):
    """
    Calculates the total average system prices, by taking into account the cost of subsidized & unsibsidized power
    to return an 'actual' system electricity price
    """
    # unpack market tables
    market_results_resampled_mean,market_results_resampled_aggregated,agent_results_resampled_mean,agent_results_resampled_aggregated = market_tables
    # calculate the system costs if smp was the price paid for everything (system_cost = smp * demand)
    # but first remove subsidized power from demand. this is biomass cofiring, biomass, waste, solar, wind onshore & wind offshore for netherlands
    unsubsidized_power = market_results.loc[:,'demand'] - (market_results.loc[:,'biomass_cofiring']+market_results.loc[:,'biomass']+ \
                            market_results.loc[:,'waste']+market_results.loc[:,'solar']+market_results.loc[:,'wind_onshore']+market_results.loc[:,'wind_offshore'])
    hourly_system_cost = unsubsidized_power.multiply(market_results.loc[:,'calculated_smp'])

    # and now calculate the actual cost for the subsidized power and add it to the total one by one
    # get the subsidized & unsubsidized parts of generation
    # for thermal, we can aggregate the power & capacities as they are handled in the same way by subsidies
    thermal_res = ['waste','biomass','chp']
    # first get the availabilities
    thermal_installed_capacities_names = agent_results_resampled_aggregated.loc(axis=1)[:,thermal_res,'power'].columns.get_level_values(0)
    thermal_installed_capacities = market.generator_long_term_availability.loc[:,thermal_installed_capacities_names]
    thermal_installed_capacities = thermal_installed_capacities.loc[thermal_installed_capacities.index.isin(market.valid_years_to_model)]
    # then find out the subsidized & unsubsidized capacities as a % of total
    subsidized_capacities_percents_thermal = calculate_subsidized_percentages(market.valid_years_to_model,cost_benefit_calculation_data.loc['biomass_cofiring_res_subsidy_start','value'],cost_benefit_calculation_data.loc['thermal_res_subsidy_duration','value'],cost_benefit_calculation_data.loc['all_subsidies_expiry','value'],thermal_installed_capacities)
    # get the generation totals
    plant_generation_thermal_res = market_results.loc[:,market_results.columns.isin(thermal_res)].sum(axis=1).copy()
    plant_generation_thermal_res.index = [date.year for date in plant_generation_thermal_res.index]
    # and multiply with subsidized_capacities_percents to get the subsidized & unsubsidized generation part
    subsidized_thermal_power = subsidized_capacities_percents_thermal.loc[:plant_generation_thermal_res.index[-1]].multiply(plant_generation_thermal_res,axis=0).fillna(0)

    # for co-firing, get generation
    cofiring_generation = market_results_resampled_aggregated.biomass_cofiring.resample('A').sum()
    cofiring_generation.index = [date.year for date in cofiring_generation.index]
    # if a limit exists, adjust the percents correctly using a cofiring_percent_adjustment percentile variable to not allow more cofiring than described
    if cost_benefit_calculation_data.loc['biomass_cofiring_res_max_yearly_generation','value'] >= 0:
        cofiring_percent_adjustment = (cost_benefit_calculation_data.loc['biomass_cofiring_res_max_yearly_generation','value']/cofiring_generation).clip(0,1)
        # apply the adjustments
        cofiring_generation = cofiring_generation*cofiring_percent_adjustment
    # if no cofiring takes place after subsidy expiry, adjust the generation further as needed
    if cost_benefit_calculation_data.loc['biomass_cofiring_res_max_yearly_generation_after_subsidy_expiry','value'] >= 0:
        # get the subsidized percentages for cofiring and reduce the generation if needed
        subsidized_capacities_percents_cofiring = calculate_subsidized_percentages(market.valid_years_to_model,cost_benefit_calculation_data.loc['biomass_cofiring_res_subsidy_start','value'],cost_benefit_calculation_data.loc['biomass_cofiring_res_subsidy_duration','value'],cost_benefit_calculation_data.loc['all_subsidies_expiry','value'],cofiring_generation.to_frame())
        cofiring_generation = cofiring_generation*subsidized_capacities_percents_cofiring.loc[:,'subsidized']
    # also, if subsidies are limited to a certain level of production, get the percentile value of that
    cofiring_subsidy_adjustment = pandas.Series(1,index=numpy.arange(cofiring_generation.index[0],cofiring_generation.index[-1]+1))
    if cost_benefit_calculation_data.loc['biomass_cofiring_res_max_yearly_generation_subsidies','value'] >= 0:
        cofiring_subsidy_adjustment = (cost_benefit_calculation_data.loc['biomass_cofiring_res_max_yearly_generation_subsidies','value']/cofiring_generation).clip(0,1)
    subsidized_capacities_percents_cofiring = calculate_subsidized_percentages(market.valid_years_to_model,cost_benefit_calculation_data.loc['biomass_cofiring_res_subsidy_start','value'],cost_benefit_calculation_data.loc['biomass_cofiring_res_subsidy_duration','value'],cost_benefit_calculation_data.loc['all_subsidies_expiry','value'],market.generators_biomass_cofiring)
    subsidized_cofiring_power = subsidized_capacities_percents_cofiring.multiply(cofiring_generation,axis=0).fillna(0)

    # and do the same for the three intermittent res
    # get the generation summed. This is recalculated, so as to also take into account the power used in the exports plant
    subsidized_capacities_percents_pv = calculate_subsidized_percentages(market.valid_years_to_model,cost_benefit_calculation_data.loc['solar_res_subsidy_start','value'],cost_benefit_calculation_data.loc['solar_res_subsidy_duration','value'],cost_benefit_calculation_data.loc['all_subsidies_expiry','value'],market.fuel.pv_growth)
    solar_generation = market_results.loc[:,'solar']
    solar_generation.index = [date.year for date in solar_generation.index]
    subsidized_solar_parks_power = subsidized_capacities_percents_pv.loc[:solar_generation.index[-1]].multiply(solar_generation,axis=0).fillna(0)

    w_onshore_generation = market_results.loc[:,'wind_onshore']
    w_onshore_generation.index = [date.year for date in w_onshore_generation.index]
    subsidized_capacities_percents_w_onshore = calculate_subsidized_percentages(market.valid_years_to_model,cost_benefit_calculation_data.loc['wind_onshore_res_subsidy_start','value'],cost_benefit_calculation_data.loc['wind_onshore_res_subsidy_duration','value'],cost_benefit_calculation_data.loc['all_subsidies_expiry','value'],market.fuel.wind_onshore_growth)
    subsidized_w_onshore_power = subsidized_capacities_percents_w_onshore.loc[:w_onshore_generation.index[-1]].multiply(w_onshore_generation,axis=0).fillna(0)

    w_offshore_generation = market_results.loc[:,'wind_offshore']
    w_offshore_generation.index = [date.year for date in w_offshore_generation.index]
    subsidized_capacities_percents_w_offshore = calculate_subsidized_percentages(market.valid_years_to_model,cost_benefit_calculation_data.loc['wind_offshore_res_subsidy_start','value'],cost_benefit_calculation_data.loc['wind_offshore_res_subsidy_duration','value'],cost_benefit_calculation_data.loc['all_subsidies_expiry','value'],market.fuel.wind_offshore_growth)
    subsidized_w_offshore_power = subsidized_capacities_percents_w_offshore.loc[:w_offshore_generation.index[-1]].multiply(w_offshore_generation,axis=0).fillna(0)

    # get their costs
    thermal_costs = pandas.concat(calculate_subsidized_power_total_cost(subsidized_thermal_power,cost_benefit_calculation_data,'thermal'),axis=1).sum(axis=1)
    biomass_cofiring_costs = pandas.concat(calculate_subsidized_power_total_cost(subsidized_thermal_power,cost_benefit_calculation_data,'biomass_cofiring'),axis=1).sum(axis=1)
    solar_parks_costs = pandas.concat(calculate_subsidized_power_total_cost(subsidized_solar_parks_power,cost_benefit_calculation_data,'solar'),axis=1).sum(axis=1)
    wind_onshore_costs = pandas.concat(calculate_subsidized_power_total_cost(subsidized_w_onshore_power,cost_benefit_calculation_data,'wind_onshore'),axis=1).sum(axis=1)
    wind_offshore_costs = pandas.concat(calculate_subsidized_power_total_cost(subsidized_w_offshore_power,cost_benefit_calculation_data,'wind_offshore'),axis=1).sum(axis=1)

    # and add them to system costs
    subsidized_costs = thermal_costs + biomass_cofiring_costs + solar_parks_costs + wind_onshore_costs + wind_offshore_costs
    hourly_system_cost += subsidized_costs.loc[:hourly_system_cost.index[-1].year].values
    # finally divide by demand to find the hourly smp
    actual_hourly_smp = hourly_system_cost/market_results.loc[:,'demand']
    return actual_hourly_smp
