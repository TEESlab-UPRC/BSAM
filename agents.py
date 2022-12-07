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
This is the main agent module, containing an instance of an agent.
Each agent is assumed to own exatly one electricity-generating plant.
Agent behaviour is needed only for non-res plants, since wind & pv agents do not take any
actions due to FIT or prices tied to the SMP.
Hydro plants do not take any actions either as the water price for a plant is tied to
the water available and not related on the agent behaviour, but hydro plant owners
are modelled as agents that don't take any actions.
Virtual plants are modelled similarly to hydro plants regarding agent behaviour
"""

import dataio
import numpy
import datetime
import pandas

class Single_Plant_Agent:
    """
    This class represents a plant owner and models his behaviour
    A plant owner primarily makes a bid to the day-ahead market each day, consisting of
    the power he can inject to the system and its price. For thermal plant owners
    (the only ones who can actually decide on a price independently in Greece,
    thus the only ones modelled) it makes no sense to not offer the max power available,
    so the power offered is always the max, while the decision only refers to the price offered.
    """
    def __init__(self, agent_actions_data_path, agent_data_datapath, plant, plant_closing_data,
                    demand, learning_algorithm,\
                    lspi_policy_path, load_learner_policies, market_data, verbosity):
        """
        At init load the possible actions and initialize the proper learner/deciding algorithm,
        overloading the self.learner module to point to the specified one (lspi, dqn, or roth-erev)
        """
        # output action is decision for the profit margin.
        # load the available margins. each action is valid for a whole day
        actions_data = dataio.load_dataframe_from_csv(agent_actions_data_path)
        # actions for non peaking generators
        if plant.kind not in ['nat-gas-occgt']:
            self.profit_margin_steps = actions_data.loc['profit_factor']
        # actions for peakers - those assume higher profit margins
        else:
            self.profit_margin_steps = actions_data.loc['profit_factor_peakers']
        self.plant = plant
        # this data contains the years back to check and the threshold as a % when deciding if the plant must close down
        self.plant_closing_data = plant_closing_data
        # this penalty is applied to calculated profit as a percentage
        # this is used for learning purposes and to simulate profit losses due to purposes other than unitcommit
        self.profit_penalty = 0
        # the verbosity function is used to enable/disable specific messages
        self.verbosity = verbosity

        # find the exact possible actions per step
        # right now this is only a function of the desired profit margins
        # later it can be a product of different actions
        self.actions_list = self.profit_margin_steps

        # load the "other data" dataframe for details regarding market rules
        self.market_data = market_data
        # and load the agent data used in all learners
        self.agent_data = dataio.load_dataframe_from_csv(agent_data_datapath)

        # this is a dataframe saving the actions of agents, the exploit status & their reward for each day
        self.taken_actions = pandas.DataFrame(columns=['action','exploit_status','profit'])
        # initialize the last_action depending on agent kind to pass to the learner
        last_action = 0
        if plant.kind not in ['hydro','virtual','imports'] and self.plant.must_run <1:
            last_action = numpy.random.randint(len(self.actions_list))

        # update the plant data with that action
        self.update_plant_data(last_action)
        self.learning_algorithm = learning_algorithm
        # get and save the policy path as an attribute
        self.model_path = ''
        if self.learning_algorithm == 'lspi':
            self.model_path = lspi_policy_path
        # load the learning algorithm we want to use
        # first get the samples count up to now
        samples_count = 0
        # also retrieve the samples count if needed
        if load_learner_policies:
            sample_count_path = self.model_path + 'sample_count' + '.npy'
            samples_count = dataio.load_numpy_array(sample_count_path)
            samples_count = samples_count[0]
        if self.learning_algorithm == 'lspi':
            import lspi
            # if we got a saved lspi policy load it. else create a new policy (lspi_policy=false)
            lspi_policy = False
            if load_learner_policies:
                # if lspi_policy_path or lspi_sample_count_path are not pointing to a file,
                # false will be returned
                lspi_policy_path = self.model_path + self.plant.name + '.npy'
                lspi_policy = dataio.load_numpy_array(lspi_policy_path)
            # our learner is the lspi object
            self.learner = lspi.LSPI(self.agent_data, self.actions_list, self.plant, lspi_policy, last_action, samples_count, self.verbosity)
        else:
            # this means no learner is used - everyone working at the MC - using a dummy learner provided by this module
            self.learner = No_Learner(samples_count)

    def closing_down_decision(self,last_date,valid_years_to_model):
        """
        Decides if the agent's plant should close down.
        This is meant to get checked once a year and take into account the tally of the last years.
        The years to take into account as well as the threshold (in % of working at max power during the specified period) are contained in the self.plant_closing_data list
        The threshold is just a % of working hours within the last years scaled to profit = 0
        """
        closing_down = False
        years_back_to_check,closing_down_threshold = self.plant_closing_data
        # find the init date to check. to do this go as far back as needed in order to stay within modelled years
        actual_years_to_check = []
        for year_count in range(years_back_to_check):
            candidate_year = last_date.year - year_count
            if candidate_year in valid_years_to_model:
                actual_years_to_check.append(candidate_year)
        # make sure that we have enough data to decide
        if actual_years_to_check and datetime.date(actual_years_to_check[-1],1,1) >= self.plant.saved_online_data.index[0][0].date():
            # for the years we need to check,
            # find the closing_down_threshold for needed energy produced during the specified timeperiod
            # and also find the scaled produced power so that it will be easy to check the closing down criterion afterwards
            closing_down_total_days = 0
            total_produced_energy_scaled = 0
            for year in actual_years_to_check:
                # add the days in the current year to the closing_down_total_days
                closing_down_total_days += (datetime.date(year,12,31) - datetime.date(year+1,1,1)).days
                saved_online_data_for_period = self.plant.saved_online_data.loc[datetime.datetime(year,1,1):datetime.datetime(year,12,31),:]
                # scale the generated power by a factor of the profit and sum the scaled generated power and added to the total
                total_produced_energy_scaled += (saved_online_data_for_period.loc[:,'power'] * saved_online_data_for_period.loc[:,'profit_factor']).sum()
            # find the actual threshold in kWh
            closing_down_threshold = closing_down_threshold * self.plant.pmax * closing_down_total_days * 24
            # do the check and return
            if total_produced_energy_scaled < closing_down_threshold: closing_down = True
        return closing_down

    def calculate_profit(self, electricity_price):
        """
        Calculate the daily profit of an action by detracting this day's income from the running costs of the plant.
        Also apply any profit penalties as needed.
        """
        cost = self.plant.calculate_one_day_cost()
        income = self.plant.calculate_one_day_income(electricity_price)
        profit = income - cost
        penalty = profit * self.profit_penalty/100
        profit -= penalty
        return profit

    def update_plant_data(self,last_action=-1):
        """
        Update the plants data as a result of the action chosen. Max price for energy is fixed by the market, thus profit may need to be adjusted.
        Last action is -1 for all situations other than init. At init the last action is found in another manner
        """
        # check if the profit factor is valid and modify if it is not, choosing a lower profit from the actions list.
        if last_action < 0:
            action_index = self.learner.current_action[0]
        else:
            action_index = last_action
        self.plant.profit_factor = self.actions_list[action_index]
        while self.plant.calculate_cost_metric('income') > self.market_data.loc['max_electricity_price','data1'] and self.plant.kind not in ['nat-gas-occgt']:
            action_index -= 1
            profit_factor = self.actions_list[action_index]
            self.plant.profit_factor = profit_factor
            if action_index < 0:
                print ('Warning!! It seems there is no action that can produce a value/kWh legal within the market rules')
                print ('Setting profit_factor to 1 (profit = 0)')
                self.plant.profit_factor = 1

    def choose_new_action(self, reward, last_state, current_state):
        """
        Choose a course of action and also apply this action to the plant == make a bid
        """
        # for hydro & virtual plants & imports, we do not need to save the action as it is always index 0
        # same is for must_run plants, since those have fixed electricity price
        if self.plant.kind not in ['hydro', 'virtual', 'imports'] and self.plant.must_run <1:
            # use learner to choose a new action
            self.learner.choose_action(reward, last_state, current_state)
            # now update the plant data
            self.update_plant_data()

    def save_learner_policy(self):
        """
        Save the current policy
        """
        if self.learning_algorithm == 'lspi':
            lspi_policy_path = self.model_path + self.plant.name+'.npy'
            dataio.save_numpy_array(lspi_policy_path, self.learner.weights)

    def save_sample_count(self):
        """
        Save the lspi samples number
        """
        # to do it in the same style as the weights (policy), make it into an array & save it in the specific model folder
        sample_count_path = self.model_path + 'sample_count' + '.npy'
        dataio.save_numpy_array(sample_count_path, numpy.array([self.learner.samples_count]))


class No_Learner:
    """
    This class is a dummy learner, that always chooses action 0 (variable cost)
    """
    def __init__(self, samples_count):
        """
        Initialize all dummy attributes
        """
        self.current_action = [0,0]
        self.samples_count = samples_count

    def choose_action(self, reward, last_state=False, current_state=False):
        """
        Dummy function that does nothing. Current action will always be 0
        """
        pass
