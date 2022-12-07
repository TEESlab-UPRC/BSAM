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
This module contains class templates (usually used as data containers)
"""
import pandas

class Plant_Commitment:
    """
    This class represents a plant commitment/dispatch of one optimization period - which is ultimately a dataframe
    And includes various related functions
    """

    def __init__(self, plants, periods):
        """"
        Initialize empty
        """
        self.commitment_values = pandas.DataFrame(0,index=periods,columns=[plant.name for plant in plants])
        self.commitment_power = pandas.DataFrame(0,index=periods,columns=["pmin","pmax"])
        self.smp = 0

    def add_plant(self, periods, plant):
        """
        Adds a plant to the commitment as online for the periods given
        """
        for period in periods:
            if self.commitment_values.loc[period,plant.name] == 0:
                self.commitment_values.loc[period,plant.name] = 1
                self.commitment_power.loc[period,"pmin"] += plant.available_min_power[period]
                self.commitment_power.loc[period,"pmax"] += plant.available_power[period]
            else:
                print ("Tried to add a plant that is already online. Starting debugger")
                import ipdb;ipdb.set_trace()

    def remove_plant(self, periods, plant):
        """
        Removes a plant from the commitment for the periods given
        """
        for period in periods:
            if self.commitment_values.loc[period,plant.name] == 1:
                self.commitment_values.loc[period,plant.name] = 0
                self.commitment_power.loc[period,"pmin"] -= plant.available_min_power[period]
                self.commitment_power.loc[period,"pmax"] -= plant.available_power[period]
            else:
                print ("Tried to remove a plant that is off. Starting debugger")
                import ipdb;ipdb.set_trace()

    def commitment_hash(self):
        """
        Returns a hash of this commitment, for use in calculations
        The hash is simply the int of the joined flattened commitment_values
        """
        # return "".join(str(i) for i in self.commitment_values.values.flatten()) # slower
        return "".join(map(str,self.commitment_values.values.flatten())) # faster

class Sample:
    """
    This is a class representing a sample (s, a, r, s'),
    where s,a,s' are index numbers and r is the reward.
    It is used as a data container.
    __eq__ (==) and __hash__ are overloaded to make for easy equality & id checking
    """
    def __init__(self, state, action, reward, nextstate):
        self.state = state
        self.action = action
        self.reward = reward
        self.nextstate = nextstate

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
            and self.__dict__ == other.__dict__)

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))
