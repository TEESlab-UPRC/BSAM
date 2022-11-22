"""
This module Defines classes that implement the Roth-Erev reinforcement learning method.
The original Roth-Erev reinforcement learning algorithm was presented by A. Roth and I. Erev in:
A. E. Roth, I. Erev, D. Fudenberg, J. Kagel, J. Emilie and R. X. Xing,
"Learning in Extensive-Form Games: Experimental Data and Simple Dynamic
Models in the Intermediate Term", Games and Economic Behavior, 8-1, pp 164-212, 1995

Erev, Ido and Roth, Alvin E., "Predicting How People Play Games:
Reinforcement Learning in Experimental Games with Unique, Mixed Strategy
Equilibria", The American Economic Review, 88-4, pp 848--881, 1998

The variation was presented in:
James Nicolaisen, Valentin Petrov, and Leigh Tesfatsion, "Market Power and Efficiency
in a Computational Electricity Market with Discriminatory Double-Auction Pricing,"
IEEE Transactions on Evolutionary Computation, Volume 5, Number 5, 2001, 504-523.
"""


import random
import numpy

class VariantRothErev():
    """
    This class represents a decision making algorithm for agents.
    It is a variant of the roth-erev algorithm and contains the needed
    functions for the algorithm to work.
    """
    def __init__(self, agent_data, actions_list, propensities, current_action_index, samples_count):
        """
        self.experimentation: The tendency for experimentation among action choices.
        The algorithm will sometimes choose non-optimal actions in favour of exploring the domain.

        Note: Be careful not to choose an experimentation value where (1-e) == e / (N - 1),
        where N is the size of the action domain (i.e. e == 0.75 and N == 4)
        this translates to N = e/(1-e) + 1
        This will result in all action propensities receiving the same
        experience update value, regardless of the last action chosen.
        Action choice probabilities will then remain uniform and no learning will occur.

        self.recency: The degree to which actions are 'forgotten'. Used to degrade the
        propensity for choosing actions. Meant to make recent experience
        more prominent than past experience in the action choice process.

        self.actions_list: Here are the possible actions for the agent
        """
        self.samples_count = samples_count
        self.experimentation = float(agent_data.loc['roth_erev_experimentation','data1'])
        assert 0.0 <= self.experimentation <= 1.0, "experimentation value out of bounds"
        self.recency = float(agent_data.loc['roth_erev_recency','data1'])
        assert 0.0 <= self.recency <= 1.0, "recency value out of bounds"
        self.actions_list = actions_list

        # initialize propensities at startup, if the propensities are false, or load them if saved
        if type(propensities) is bool and propensities == False:
            self.propensities = self.reset_propensities()
        else:
            self.propensities = propensities
        # ensure experimentation value is ok
        assert not numpy.isclose(len(self.actions_list), (self.experimentation/(1-self.experimentation) + 1)), \
                    "Error! Combination of roth-erev experimentation and discrete actions means no learning is possible"
        # here the current action is saved
        self.current_action = [current_action_index,1]

    def reset_propensities(self):
        """
        This function will reset propensities to conform with a uniform distribution
        """
        discrete_actions = len(self.actions_list)
        return numpy.full(discrete_actions, 1/discrete_actions, float)

    def update_propensities(self, reward):
        """
        Updates the propensities for all actions. The propensity for last
        action chosen will be updated using the feedback value that resulted
        from performing the action.

        If j is the index of the last action chosen, r_j is the reward received
        for performing j, i is the current action being updated, q_i is the
        propensity for i, and self.recency is the recency parameter, then this update
        function can be expressed as::  q_i+1 = (1-phi) * q_i + E(i, r_j)
        """
        for action_index, action in enumerate(self.actions_list):
            # the carryover is a direct product of the recency and propensity of the last 'round'
            carry_over = (1 - self.recency) * self.propensities[action_index]
            # experience is affected by the last action and the reward
            experience = self.get_new_experience_value(action_index, action, self.current_action[0], reward)
            # a propensity cannot be below zero:
            propensity = carry_over+experience
            if propensity < 0: propensity = 0
            # now update
            self.propensities[action_index] = propensity
        # and update the samples count since we used a sample
        self.samples_count += 1

    def get_new_experience_value(self, action_index, current_action, last_action, reward):
        """
        In this variation propensities for all actions are updated and
        similarity does not come into play. If the actionIndex points to the
        action the reward is associated with (usually the last action taken)
        then simply adjust the weight by the experimentation. Otherwise increase
        the weight of the action by a small portion of its current propensity.

        If j is the index of the last action chosen, r_j is the reward received
        for performing j, i is the current action being updated, q_i is the
        propensity for i, n is the size of the action domain and e is the
        experimentation parameter, then this experience function can be
        expressed as::
                        |-->  r_j * (1-e)         if i = j
            E(i, r_j) = |
                        |--> q_i * (e /(n-1))    if i != j
        """

        if current_action == last_action:
            experience = reward * (1 - self.experimentation)
        else:
            propensity = self.propensities[action_index]
            experience = propensity * (self.experimentation / (len(self.actions_list) - 1))
        return experience


    def choose_action(self, reward, last_state=False, current_state=False):
        """
        Update propensities, calculate probabilities and choose one action.
        The "last_state, current_state" variables are not used and are just passed to ensure api compatibility with other learners
        """
        probabilities = []
        self.update_propensities(reward)
        summedProps = sum(self.propensities)
        for propensity in self.propensities:
            probability = propensity/summedProps
            probabilities.append(probability)

        # get the index of the next action randomly based on the probabilities to select it
        action = numpy.random.choice(len(self.actions_list), p=probabilities)
        self.current_action = [action,1]
