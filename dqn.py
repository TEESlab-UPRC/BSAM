import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32"
import keras
import numpy
import class_library


class Dqn:
    """
    this is an alternate learner module using a deep neural network implementation to model agent learning
    Each agent has his own instance if this class
    """
    def __init__(self, agent_data, possible_actions, plant, loaded_model_path, current_action_index, samples_count):
        """
        Create the dqn. and save any other needed variables
        """

        # also use the other specified options
        self.agent_data = agent_data
        self.plant = plant
        self.possible_actions = possible_actions
        self.possible_actions_num = len(self.possible_actions)
        # this is the action that was last selected by the agent (its index)
        self.current_action = [current_action_index,0]
        # sample related
        self.available_samples = []
        self.samples_count = samples_count
        self.retrain_frequency_init = int(self.agent_data.loc['dqn_update_frequency','data1'])
        self.retrain_frequency = int(self.agent_data.loc['dqn_update_frequency','data1'])

        # exploration step is the step with which to lower exploration factor for each sample. its linear
        self.exploration_step = (float(self.agent_data.loc['dqn_min_exploration_factor','data1']) - 1) / (int(self.agent_data.loc['dqn_samples_to_max_exploitation','data1']) - int(self.agent_data.loc['dqn_min_samples_needed','data1']))

        # if we did not load an already saved model (weights and all),
        # initialize a new one - this will get updated as we go
        # if we are instructed to load a model but it does not exist, also create a new one..
        self.initial_model_path = loaded_model_path
        if self.initial_model_path is False or not os.path.exists(self.initial_model_path):
            self.model = self.create_model()
        else:
            # if we did load a saved model, use it and set the samples to a large enough
            # amount so that there will be no initial exploration - assuming 300 samples are ok
            self.model = keras.models.load_model(self.initial_model_path)

    def create_model(self):
        """
        Create a dqn to get used for learning. Inputs, outputs and layers are defined from dqn data
        """
        # A state is described by 3 variables (right now demand,nat gas price, smp), with the action being the 4th, thus input vector is of sixe 4
        # While the output is only the expected profit, thus of size 1
        model = keras.models.Sequential()
        # load general layer data - this could be specialized of course
        layer_size = int(self.agent_data.loc['dqn_layer_size','data1'])
        layer_kernel_init_type = self.agent_data.loc['dqn_layer_kernel_init_type','data1']
        layer_bias_init_type = self.agent_data.loc['dqn_layer_bias_init_type','data1']
        layer_activation_type = self.agent_data.loc['dqn_layer_activation_type','data1']
        dropout_frequency = float(self.agent_data.loc['dqn_dropout_frequency','data1'])
        # create & add the input layer & hidden layer 1
        input_layer_input_nodes = int(self.agent_data.loc['dqn_input_layer_input_nodes','data1'])
        model.add(keras.layers.Dense(layer_size, activation=layer_activation_type, input_shape=(input_layer_input_nodes,), kernel_initializer=layer_kernel_init_type, bias_initializer=layer_bias_init_type))
        # create & add all the middle layers
        for hidden_layer in range(int(self.agent_data.loc['dqn_hidden_layers_number','data1'])):
            new_layer = keras.layers.Dense(layer_size, activation=layer_activation_type, kernel_initializer=layer_kernel_init_type, bias_initializer=layer_bias_init_type)
            model.add(new_layer)
            # also add a dropout after each layer to avoid overlearning
            model.add(keras.layers.Dropout(dropout_frequency))
        # create & add the output layer - no activation will be used for it & the layer will have size equal to the possible actions, so each output represents a Qvalue of an action
        output_layer_size = self.possible_actions_num
        model.add(keras.layers.Dense(output_layer_size, kernel_initializer=layer_kernel_init_type, bias_initializer=layer_bias_init_type))

        # compile the model
        # compile uses theano/tensorflow as backend & these auto choose CPU/GPU or both
        # At compile we need to specify the loss function to use to evaluate a set of weights, the optimizer to search through different weights & optional metrics we would like to collect and report during training.
        # used logarithmic loss for binary classification problem (=binary_crossentropy) | gradient descend optimized (=adam) by default
        # Finally, because it is a classification problem, we will collect and report the classification accuracy as the metric.
        # for dqn, loss needs to be L = {[reward + gamma * max[Q(s',a')]] - Q(s,a)}^2, where reward = reward, gamma = discount factor, Q(s',a') = Q-val of the best action, Q(s,a) = Q-val of the current action
        loss_type = self.agent_data.loc['dqn_loss_type','data1']
        optimizer_type = self.agent_data.loc['dqn_optimizer_type','data1']
        model.compile(loss=loss_type, optimizer=optimizer_type, metrics=['accuracy'])
        return model


    def train_model(self):
        """
        This trains the model using the saved samples as inputs & outputs
        """
        # We can train or fit our model on our loaded data by calling the fit() function on the model.
        # The training process will run for a fixed number of iterations through the dataset called epochs, that we must specify using the nb_epoch argument.
        # We can also set the number of instances that are evaluated before a weight update in the network is performed, called the batch size and set using the batch_size argument.
        train_batch_size = int(self.agent_data.loc['dqn_train_batch_size','data1'])
        train_epochs = int(self.agent_data.loc['dqn_train_epochs','data1'])
        # populate the inputs and outputs lists
        inputs = []
        outputs = []
        # select random samples from the current ones, so as to avoid falling into local maxima
        # this means that some samples will get reused, but not within the same batch
        selected_samples = numpy.random.choice(self.available_samples, train_batch_size)
        # the fitting will happen for all the batch simultaneously and not for each sample individually
        # thus the current weights are used for creating the predicted outputs for each state
        for sample in selected_samples:
            # get the i/o for each sample
            sample_input = numpy.array([sample.state])
            # get an initial predicted output for the sample's initial state
            sample_output = self.model.predict(sample_input)
            # get the prediction for the q-values that result from the final state. the [0] is used since a [[val,val,..]] is returned by predict
            predicted_action_values = self.model.predict(numpy.array([sample.nextstate]))[0]
            best_predicted_action_value = numpy.max(predicted_action_values)
            best_predicted_action_index = numpy.argmax(predicted_action_values)
            # normalize the reward so that it will not be too high, thus that we can improve the Q-learning process
            reward = sample.reward / float(self.agent_data.loc['reward_normalization_factor','data1'])
            # and update this output at the index of the expected best action
            sample_output[0][best_predicted_action_index] = reward + float(self.agent_data.loc['dqn_discount_factor','data1']) * best_predicted_action_value
            # now append those i/o's to the list that will get used for the training
            inputs.append(sample_input[0])
            outputs.append(sample_output[0])
        # convert lists to arrays. maybe we can skip it?
        inputs = numpy.array(inputs)
        outputs = numpy.array(outputs)
        # do the fitting for the i/o's, all together. this could be done step by step too
        self.model.fit(inputs,outputs,epochs=train_epochs,batch_size=train_batch_size)
        # remove samples if the max number of samples was exceeded
        self.remove_samples_if_required()
        # also update the total samples used for fitting counter
        self.samples_count += len(self.available_samples)

    def choose_action(self, reward, last_state, new_state):
        """
        This function gets called at every turn.
        It is the 'main' function that starts the learner logic by requesting a new decision.
        Also, if the last state exists, it makes a sample and add it to the available samples list
        When these samples are numerous enough, train_model is run to update the model with the new samples
        """
        if last_state is not None:
            # a new state-reward was returned, first update the samples list
            new_sample = class_library.Sample(last_state, self.current_action[0], reward, new_state)
            self.available_samples.append(new_sample)
            # be sure to update the model if enough new samples have been collected
            if len(self.available_samples) % self.retrain_frequency == 0:
                self.train_model()
                # also update the update frequecy so that it slows down logarithmically - after 100 samples & for every minibatch
                if self.samples_count > 100:
                    self.retrain_frequency = round(numpy.log10(self.samples_count)*self.retrain_frequency_init)

        # find the best action depending on find_best_action_mode and save it in the agent module, also find the exploit_status
        self.current_action = self.find_best_action(new_state)

    def find_best_action(self,current_state):
        """
        If not enough samples have been used, do not exploit but explore
        Assuming we have enough samples, linearly lower the exploration factor until
        the min exploration factor is reached at the time specified as samples_to_max_exploitation
        and it cannot be under the min_exploration_factor
        """
        if (self.samples_count > int(self.agent_data.loc['dqn_min_samples_needed','data1'])):
            exploration_factor = max(1+self.exploration_step*self.samples_count, float(self.agent_data.loc['dqn_min_exploration_factor','data1']))
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
            q_values = self.model.predict(numpy.array([current_state]))
            # get the index of the max value
            choice = numpy.argmax(q_values)
        return [choice,exploit_status]

    def remove_samples_if_required(self):
        """
        Since the samples are added sequentially, we only need to keep a slice of the samples list
        Deselecting as many as needed from the start
        """
        current_samples = len(self.available_samples)
        max_samples = int(self.agent_data.loc['dqn_max_samples_allowed','data1'])
        if current_samples > max_samples:
            samples_to_remove = current_samples - max_samples
            self.available_samples = self.available_samples[samples_to_remove:]
