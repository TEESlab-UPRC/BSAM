"""
This module is used to conduct cost-benefit calculations for the greek electricity market based on current subsidies framework & res projections
"""

import numpy
import pandas
import dataio
import datetime

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
    subsidy_start_year = int(subsidy_start_year)
    subsidy_duration = int(subsidy_duration)
    subsidy_expiry_year = int(subsidy_expiry_year)
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

def calculate_subsidized_power_total_cost(generated_power,cost_benefit_calculation_data,tax_type,market):
    """
    Calculates the total cost of the subsidy, given the per unit cost
    """
    if cost_benefit_calculation_data.loc[tax_type+'_res_subsidy_price_mwh','value'] == 'smp':
        # get the yeatly subsidy price per mwh & use it to get the yearly cost
        smp_prices = market.fuel.calculated_smp.loc[:,'daily_smp']
        unsubsidized_cost = pandas.DataFrame(0,index=smp_prices.index,columns=['unsubsidized_cost'])
        subsidized_cost = pandas.DataFrame(0,index=smp_prices.index,columns=['subsidized_cost'])
        generated_power.index = smp_prices.index
        for date in smp_prices.index:
            subsidy_price = float(cost_benefit_calculation_data.loc[tax_type+'_res_subsidy_price_mwh',str(date.year)])
            if subsidy_price < 0:
                subsidy_price = smp_prices.loc[date]*abs(subsidy_price)
            subsidized_cost.loc[date] = subsidy_price * generated_power.loc[date,'subsidized']
            unsubsidized_cost.loc[date] = smp_prices.loc[date] * generated_power.loc[date,'unsubsidized']
    else:
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
    # this function is broken!!!!
    # unpack market tables
    market_results_resampled_mean,market_results_resampled_aggregated,agent_results_resampled_mean,agent_results_resampled_aggregated = market_tables
    # calculate the system costs if smp was the price paid for everything (system_cost = smp * demand)
    # the subsidized res types are set in cost_benefit_calculation_data
    subsidized_generation_types = cost_benefit_calculation_data.loc['subsidized_generation_types',:].dropna().tolist()
    subsidized_costs = pandas.Series(0,index=market.fuel.calculated_smp.index)
    unsubsidized_power = market_results.loc[:,'demand']
    # for each subsidized type, get the total, subsidized and unsubsidized amounts and adjust demand to ommit subsidies (demand will be used to calculate the remunaration via smp)
    for generation_type in subsidized_generation_types:
        if generation_type in market_results.columns and market_results.loc[:,generation_type].sum() > 0:
            # first get the availabilities
            # if we are calculating for thermal res, do the aggregation across all thermal types
            if generation_type == 'thermal':
                generation_installed_capacities_names = agent_results_resampled_aggregated.loc(axis=1)[:,cost_benefit_calculation_data.loc['thermal_res',:].values,'power'].columns.get_level_values(0)
                generation_installed_capacities = market.generator_long_term_availability.loc[:,generation_installed_capacities_names]
                generation_installed_capacities = generation_installed_capacities.loc[generation_installed_capacities.index.isin(market.valid_years_to_model)]
                plant_generation = market_results.loc[:,generation_type].dropna(axis=1).sum(axis=1).copy()
                plant_generation.index = [date.year for date in plant_generation.index]
                subsidized_capacities_percents = calculate_subsidized_percentages(market.valid_years_to_model,cost_benefit_calculation_data.loc[generation_type+'_res_subsidy_start','value'],cost_benefit_calculation_data.loc[generation_type+'_res_subsidy_duration','value'],cost_benefit_calculation_data.loc['all_subsidies_expiry','value'],generation_installed_capacities)
                subsidized_power = subsidized_capacities_percents.loc[:plant_generation.index[-1]].multiply(plant_generation,axis=0).fillna(0)
            # if we are calculating for cofiring, respect any upper generation limits as set
            elif generation_type == 'biomass_cofiring_res':
                plant_generation = market_results_resampled_aggregated.biomass_cofiring.resample('A').sum()
                plant_generation.index = [date.year for date in cofiring_generation.index]
                # if a limit exists, adjust the percents correctly using a cofiring_percent_adjustment percentile variable to not allow more cofiring than described
                if cost_benefit_calculation_data.loc[generation_type+'_res_max_yearly_generation','value'] >= 0:
                    percent_adjustment = (cost_benefit_calculation_data.loc[generation_type+'_max_yearly_generation','value']/plant_generation).clip(0,1)
                    # apply the adjustments
                    plant_generation = plant_generation*cofiring_percent_adjustment
                # if no cofiring takes place after subsidy expiry, adjust the generation further as needed
                if cost_benefit_calculation_data.loc[generation_type+'_res_max_yearly_generation_after_subsidy_expiry','value'] >= 0:
                    # get the subsidized percentages for cofiring and reduce the generation if needed
                    subsidized_capacities_percents_cofiring = calculate_subsidized_percentages(market.valid_years_to_model,cost_benefit_calculation_data.loc[generation_type+'_res_subsidy_start','value'],cost_benefit_calculation_data.loc[generation_type+'_res_subsidy_duration','value'],cost_benefit_calculation_data.loc['all_subsidies_expiry','value'],plant_generation.to_frame())
                    plant_generation = plant_generation*subsidized_capacities_percents_cofiring.loc[:,'subsidized']
                # also, if subsidies are limited to a certain level of production, get the percentile value of that
                cofiring_subsidy_adjustment = pandas.Series(1,index=numpy.arange(cofiring_generation.index[0],cofiring_generation.index[-1]+1))
                if cost_benefit_calculation_data.loc[generation_type+'_res_max_yearly_generation_subsidies','value'] >= 0:
                    cofiring_subsidy_adjustment = (cost_benefit_calculation_data.loc[generation_type+'_res_max_yearly_generation_subsidies','value']/cofiring_generation).clip(0,1)
                subsidized_capacities_percents = calculate_subsidized_percentages(market.valid_years_to_model,cost_benefit_calculation_data.loc[generation_type+'_res_subsidy_start','value'],cost_benefit_calculation_data.loc[generation_type+'_res_subsidy_duration','value'],cost_benefit_calculation_data.loc['all_subsidies_expiry','value'],market.generators_biomass_cofiring)
                subsidized_power = subsidized_capacities_percents.multiply(cofiring_generation,axis=0).fillna(0)
            # for intermittent res
            else:
                # get the subsidy percentages
                if generation_type == 'solar':
                    subsidized_capacities_percents = calculate_subsidized_percentages(market.valid_years_to_model,cost_benefit_calculation_data.loc[generation_type+'_res_subsidy_start','value'],cost_benefit_calculation_data.loc[generation_type+'_res_subsidy_duration','value'],cost_benefit_calculation_data.loc['all_subsidies_expiry','value'],market.fuel.pv_growth)
                elif generation_type == 'wind_onshore':
                    subsidized_capacities_percents = calculate_subsidized_percentages(market.valid_years_to_model,cost_benefit_calculation_data.loc[generation_type+'_res_subsidy_start','value'],cost_benefit_calculation_data.loc[generation_type+'_res_subsidy_duration','value'],cost_benefit_calculation_data.loc['all_subsidies_expiry','value'],market.fuel.wind_onshore_growth)
                elif generation_type == 'wind_offshore':
                    subsidized_capacities_percents = calculate_subsidized_percentages(market.valid_years_to_model,cost_benefit_calculation_data.loc[generation_type+'_res_subsidy_start','value'],cost_benefit_calculation_data.loc[generation_type+'_res_subsidy_duration','value'],cost_benefit_calculation_data.loc['all_subsidies_expiry','value'],market.fuel.wind_offshore_growth)
                # get the generation
                plant_generation = market_results.loc[:,generation_type]
                plant_generation.index = [date.year for date in plant_generation.index]
                subsidized_power = subsidized_capacities_percents.loc[plant_generation.index[0]:plant_generation.index[-1]].multiply(plant_generation,axis=0).fillna(0)
            # also adjust demand, removing the part of generation that will be subsidized
            unsubsidized_power = unsubsidized_power.add(-subsidized_power.sum(axis=1).values)
            # get the cost for the generation type
            costs = pandas.concat(calculate_subsidized_power_total_cost(subsidized_power,cost_benefit_calculation_data,generation_type,market),axis=1).sum(axis=1)
            subsidized_costs = subsidized_costs.add(costs.values)
    # calculate the system costs
    hourly_system_cost = unsubsidized_power.multiply(market_results.loc[:,'calculated_smp'])
    # add the subsidy costs to the system cost
    hourly_system_cost += subsidized_costs.loc[:hourly_system_cost.index[-1]].values
    # finally divide by demand to find the hourly smp
    actual_hourly_smp = hourly_system_cost/market_results.loc[:,'demand']
    return actual_hourly_smp
