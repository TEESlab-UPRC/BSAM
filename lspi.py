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
This module handles agent behaviour and decision making by using the LSPI algorithm
presented by Lagoudakis and Parr in "Least-Squares Policy Iteration", Journal of Machine Learning Research 4 (2003) 1107-1149
The implied policy is generated over the course of the decision-making process and this policy
must be able to find the best (perceived) action given any system state.
Initially the agents take actions at random (exploration) if they don't have enough information,
but this changes gradually.
LSPI works in "online" mode, meaning that as long as enough samples are collected, the
policy is updated (LSTDQ) with the information these provide.
The policies can be stored and loaded, along with the used samples number.
It is assumed that about 2 years worth of samples are needed for lspi to work well for the given
problem
"""

import numpy
import copy
import class_library


class LSPI:
    """
    This class contains all the lspi-related data and functions.
    Each agent has his own instance if this class
    """
    def __init__(self, agent_data, possible_actions, plant, lspi_policy, current_action_index, samples_count, verbosity):
        """
        Loads needed data to initialize a policy, or load an existing one.
        """
        # ----------- general -----------
        # all possible actions of the agent
        self.plant = plant
        self.possible_actions = possible_actions
        self.possible_actions_num = len(self.possible_actions)
        # at first there will be no samples.
        # if too few samples are known, the learner will only do exploration until enough have been collected
        # the unused samples is a list where any sample not yet used in lstdq will go initially
        # after getting used to improve the weights, it will become part of the samples set
        self.unused_samples=[]
        # the samples count is a counter of the samples we got.
        self.samples_count = samples_count
        self.agent_data = agent_data
        # This num is lspi update frequency. every lspi_update_frequency new samples we will run lstdq again
        # and since the frequency will change as the samples increase, bakup the initial too
        self.lspi_update_frequency_init = int(self.agent_data.loc['lspi_update_frequency','data1'])
        self.lspi_update_frequency = self.lspi_update_frequency_init
        self.verbosity = verbosity

        # ----------- state related -----------
        # state : A column array with an arbitrary description of states
        # a state is for this problem represented by a 2d vector.
        # the 2 dimensions are: total energy demand for the day,
        # and average cost/KWh. each state represents one whole day.
        # those 2 dimensions will have 10 discrete states each, thus 100 different states
        # The 2 functions will each be a row in the column array
        # & state will be a column array of 2 variables by default
        self.state_dimensions = int(self.agent_data.loc['lspi_state_dimensions','data1'])
        self.state_scale_per_dimension = self.agent_data.loc['lspi_state_scale_per_dimension'].astype(int)
        # ----------- basis functions related -----------
        # type of polynomial function to be used
        self.poly_type = int(self.agent_data.loc['lspi_poly_type','data1'])
        # degree of polynomial function to be used
        # this needs to be set higher by one than the desired since degree=1 means f = c1, degree=2 means f = c1 + c2x etc.
        self.poly_degree = int(self.agent_data.loc['lspi_poly_degree','data1'])
        # this array contains all the phi's already calculated by get basis, so that they don't need to get calculated again
        # elements inside are in the following format: [[[state,phi],...]....], where each outer index point corresponds to an action index
        # this could get exceedingly large, so it will only be used per one sample and then reset.
        # a dict would solve the above problem, but it is unlikely that a state will repeat itself, so it'd be of marginal use
        self.saved_phi = []
        for i in range(self.possible_actions_num):
            self.saved_phi.append({})
        # ----------- policy & lspi related -----------
        # policy exploration rate (float)
        self.exploration_factor_coefficient = float(self.agent_data.loc['lspi_exploration_factor','data1'])
        # policy discount factor (float)
        self.discount_factor = float(self.agent_data.loc['lspi_discount_factor','data1'])
        # this defines how the weights to be multiplied with the basis functions will be initially created
        self.init_weights = int(self.agent_data.loc['lspi_init_weights','data1'])
        # epsilon is a positive small real number used as the termination criterion.
        # LSPI converges if the distance between weights of consequtive iterations is less than epsilon.
        self.epsilon = float(self.agent_data.loc['lspi_epsilon','data1'])
        # delta is a positive real number, to be used within lstdq
        self.delta = float(self.agent_data.loc['lspi_delta','data1'])
        # An integer indicating the maximum number of LSPI iterations
        self.max_iterations = int(self.agent_data.loc['lspi_max_iterations','data1'])
        self.number_of_basis_functions = self.poly_degree * self.possible_actions_num * self.state_dimensions
        # if we did not load an already saved policy,
        # initialize a policy - this is the current LSPI policy and will get updated as we go
        # the policy is implied by the weights, but initially set a 'None' for it.
        self.initial_policy = lspi_policy
        if self.initial_policy is False:
            self.weights = self.initialize_weights()
        else:
            # if we did load a saved policy, use it and set the samples to a large enough
            # amount so that there will be no initial exploration - assuming 300 samples are ok
            self.weights = self.initial_policy

        # since we will be running online, it is highly important to remember the
        # lstdq variables so that we will be solving the problem only for the new samples added
        # and not for all the samples each time. these variables are B and b and are made
        # an attribute so as to make them persistent (global) - check the LSTDQ_opt function for more info
        # initialize B and b
        #self.lstdq_B = numpy.eye(self.number_of_basis_functions, self.number_of_basis_functions) * 1/self.delta
        self.lstdq_B = numpy.eye(self.number_of_basis_functions, self.number_of_basis_functions) * self.delta
        self.lstdq_b = numpy.zeros((self.number_of_basis_functions, 1), 'f')

        # this is the action that was last selected by the agent (its index)
        self.current_action = [current_action_index,0]

        # this is for debug purposes and could be deleted if we deem convergence is ok
        self.non_convergence = 0
        self.reward_normalization_factor = float(self.agent_data.loc['reward_normalization_factor','data1'])

    def lspi(self):
        """
        This is the least-squares policy iteration main function.
        It runs as soon as enough samples are collected and updates the policy (weights)
        by using the LSTDQ_opt_online method until the distance between two updates
        is close to zero (lspi converges) or we exceed the max iterations allowed.
        Via experimentation it was found that with max iterations set to 50,
        lspi converges at the overwhelming majority of situations for this problem.
        The policy is updated automatically in the end, so no need to return anything.
        """
        # initialize policy iteration
        iteration = 0
        distance = numpy.inf

        # main LSPI loop
        while distance > self.epsilon and self.max_iterations > iteration:

            # Evaluate the current policy (and implicitly improve)
            neweights = self.LSTDQ_opt_online()
            # compute the distance between the current and the previous policy
            difference = self.weights - neweights
            distance = numpy.linalg.norm(difference)
            # update the current policy
            self.weights = neweights.copy()
            # update iteration
            iteration += 1

        # now that the new samples have been used, clean up the unused samples list
        self.samples_count += len(self.unused_samples)
        self.unused_samples = []

        # reset saved phi afterwards, so that it will not consume major amounts of space
        self.saved_phi = []
        for i in range(self.possible_actions_num):
            self.saved_phi.append({})

        # report if lspi converged or not, for testing purposes
        if self.max_iterations > iteration:
            if self.verbosity > 0:
                print ('%s: LSPI converged after %s iterations' % (self.plant.name,iteration))
        else:
            self.non_convergence +=1
            print ('%s: LSPI finished WITHOUT convergence. Distance: %s' % (self.plant.name,distance))


    def LSTDQ_opt_online(self):
        """
        This is a performance optimization of LSTDQ as per the lagoudakis paper
        It also is modified to run "online", that is to update the policy with a few
        samples at a time
        It uses any up-to-now unused samples in order to update the policy weights.
        """
        # Loop through the samples
        # for reverse, point would be to change the 'reward'!. reward what? being close to reality?????
        # how? reward getting actions close to the real????
        for sample in self.unused_samples:
            phi = self.get_basis(sample.state, sample.action)
            # this is implicitly using the older weights = current policy
            next_action_index,exploit_status = self.find_best_action(sample.nextstate)
            nextphi = self.get_basis(sample.nextstate, next_action_index)
            next_phi_diff = (phi-self.discount_factor*nextphi).transpose()
            bphi = numpy.dot(self.lstdq_B,phi)
            bphidiff = numpy.dot(bphi,next_phi_diff)
            Bupper = numpy.dot(bphidiff,self.lstdq_B)
            Blower = 1+numpy.dot(numpy.dot(next_phi_diff,self.lstdq_B),phi)

            # if Blower becomes zero, the sample will be discarded to avoid divide-by-zero
            if Blower!=0:
                # update the B,b global lstdq variables
                self.lstdq_B -= Bupper/Blower
                self.lstdq_b += phi*sample.reward/self.reward_normalization_factor
            else:
                print ("LSTDQ: Discarding sample where Blower would be zero")

        # find the new weights
        weights = numpy.dot(self.lstdq_B,self.lstdq_b)
        return weights


    def find_best_action(self, state):
        """
        This function computes the "best" action at a given state, based on a given policy &
        returns the action of the pair (state, action)
        The 'real' exploration factor is a log function so that it will slowly become smaller
        as the samples grow in number - as of now, with exploration_factor_coefficient = 1:
        3 samples = 85%; 10 samples = 66%, 100 samples = 33%, 600 samples = 7% exploration.
        We only allow exploitation with a sample size >= 5
        """
        if (self.samples_count > 4):
            exploration_factor = max(self.exploration_factor_coefficient-self.exploration_factor_coefficient*0.145*numpy.log(self.samples_count),0.02)
        else:
            # manually set it to 2, to make certain we will do exploration with low sample size
            exploration_factor = 2
        choice = 0
        exploit_status = 0
        if (numpy.random.rand() < exploration_factor):
            # if we are exploring, or the samples are too few pick one action at random
            choice = numpy.random.choice(self.possible_actions_num)
        else:
            exploit_status = 1
            # if not, pick the action with the maximum Q-value
            best_q_value = -numpy.inf
            best_actions = []
            # find all actions with maximum Q-value
            for index in range(self.possible_actions_num):
                phi = self.get_basis(state, index)
                q_value = numpy.dot(phi.transpose(), self.weights)[0][0]
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_actions = [index]
                elif q_value == best_q_value:
                    best_actions.append(index)
            # and pick one of them
            try:
                choice = numpy.random.choice(best_actions)
            except:
                print('No action was found. Maybe the state is nan?')
                import ipdb;ipdb.set_trace()
        return [choice,exploit_status]

    def choose_action(self, reward, last_state, new_state):
        """
        This function gets called at every turn.
        It is the 'main' function that starts the lspi logic by requesting a new decision.
        Also, if the last state exists, it makes a sample and add it to the unused samples list
        When these samples are numerous enough, lspi is run to update the policy
        """
        if last_state is not None:
            # a new state-reward was returned, first update the unused samples list
            new_sample = class_library.Sample(last_state, self.current_action[0], reward, new_state)
            self.unused_samples.append(new_sample)
            # be sure to update the policy if enough new samples have been collected
            if len(self.unused_samples) % self.lspi_update_frequency == 0:
                self.lspi()
                # also update the update frequecy so that it slows down logarithmically - after 100 samples
                if self.samples_count > 100:
                    self.lspi_update_frequency = round(numpy.log10(self.samples_count)*self.lspi_update_frequency_init)

        # find the best action depending on find_best_action_mode and save it in the agent module, also find the exploit_status
        self.current_action = self.find_best_action(new_state)

    def get_basis(self, state, action_index):
        """
        As LSPI is based on a set of parametric weights, this function calculates
        creates the basis functions representing a specific state.
        It computes a set of polynomial (on "state") basis functions by taking a specific state
        and action as index
        """
        phi = self.saved_phi[action_index].get(state,[False])
        phi = [False]
        # if the phi is not known, find it
        if not phi[0]:
            # initialize
            phi = numpy.zeros((self.number_of_basis_functions, 1), 'f')

            # These variables are used to help filling out the correct places of the phi array
            # (that depend on the action, with the pertinent polynomial terms
            # (that depend on the state and polynomial degree set)
            # Step is the degree of the polynomial functions representing the process
            # Segment is the number of array cells to fill for one polynomial function
            # Base is the number of array cells where we start (1st cell of 1st function)
            step = self.poly_degree
            segment = int(self.number_of_basis_functions/self.possible_actions_num)
            base = action_index*segment

            # compute values for each function & function variable (polynomial terms)
            phi[base:(base+segment):step] = 1.0
            for i in range(step):
                for j in range(len(state)):
                    # scale each dimension of the state to [0,10), currently assuming 100GWh max daily demand
                    try:
                        # our state[j] size is always one per element in wmsim. if it were not, we'd need to use the fix below
                        phi[base+i+j*(step)] = self.create_poly([10/self.state_scale_per_dimension[j] * state[j]], i, self.poly_type)
                        #if numpy.size(state[j]) == 1:
                        #   phi[base+i+j*step] = self.create_poly([10/self.state_scale_per_dimension[j] * state[j]], i, self.poly_type)
                        #else:
                        #phi[base+i+j*step] = self.create_poly(10/self.state_scale_per_dimension[j] * state[j], i, self.poly_type)
                    except ValueError:
                        print ('something went wrong with lspi basis creation, please investigate')
                        import ipdb;ipdb.set_trace()
            # save the phi to the list
            self.saved_phi[action_index].update({state:phi})
        return phi

    def initialize_weights(self):
        """
        This function initializes the weights (policy) array depending on the init_weights specified
        weights: A column array of weights (one for each basis function)
        init_weights: -1 means random weights, 0 means zeros,
        anything else must be a real number to be multiplied with ones and fill the array
        """

        # Initial weights can be chosen in one of three ways
        if self.init_weights == -1:
            weights = numpy.random.rand(self.number_of_basis_functions,1)*numpy.ones((self.number_of_basis_functions,1), 'f')   # Random
        else:
            weights = self.init_weights*numpy.ones((self.number_of_basis_functions,1), 'f')   # Ones * init
        return weights

    def create_poly(self, state, n, t):
        """
        This function finds and returns the value of the variables of a function
        (used to represent a process), depending on an input vector representing a state.
        The type and degree of the function is specified with t,n.

        s: input vector
        n: degree of polynomial
        t: type of polynomial
        -1 Powers
        -2 Laguerre

        Default in this lspi implementation is n=0~3 (3rd degree poly); t=2 (laguerre); s.size = 1 (1-dimension per input vector)
        """

        # the original func is create_poly_bak. this is an optimized alternative for wmsim
        # our state size is always one in wmsim. if it were not, we'd need to use the fixes commented out below to find g
        c = []

        if t == 1:
            # t=1 became second place since t=2 is the default in this lspi- perfrmance opt
            g = numpy.zeros((1,1), 'f')
            N = 0
            d = 1
            c.append(1)
            g[0] = state[0]**n

        elif t == 2:
            N = n
            g = numpy.zeros((1,N+1), 'f')
            d = 1
            for m in range(N+1):
                c.append((-1)**m/numpy.math.factorial(m)*numpy.math.factorial(n)/numpy.math.factorial(m)/numpy.math.factorial(n-m))
                g[0][m] = state[0]**m
        else:
            raise ValueError('error in polynomial type')

        return d*numpy.dot(g, c)
