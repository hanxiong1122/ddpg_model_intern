"""
Use DDPG to train a stock trader based on a window of history price
"""

from __future__ import print_function, division

from model.ddpg.actor import ActorNetwork
from model.ddpg.critic import CriticNetwork
from model.ddpg.ddpg import DDPG
from model.ddpg.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

from environment.portfolio import PortfolioEnv
from utils.data import read_stock_history, normalize, read_uptodate_data, fetch_data
from utils.modelutils import  generate_train_folder, load_config, build_parser


import os
import numpy as np
import tflearn
import tensorflow as tf
import argparse
import pprint
import logging
import json
import copy
from scipy import stats

DEBUG = False



def get_variable_scope(window_length, predictor_type, use_batch_norm):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'
    return '{}_window_{}_{}'.format(predictor_type, window_length, batch_norm_str)

def activation_net(net, activation_function):
    if activation_function == "relu":
        return tflearn.activations.relu(net)
    if activation_function == "leaky_relu":
        return tflearn.activations.leaky_relu(net)
    if activation_function == "prelu":
        return tflearn.activations.prelu(net)
    if activation_function == "tanh":
        return tflearn.activations.tanh(net)
    if activation_function == "sigmoid":
        return tflearn.activations.sigmoid(net)
    if activation_function == "relu6":
        return tflearn.activations.relu6(net)
    raise NotImplementedError

def stock_predictor(inputs, feature_number, predictor_type, use_batch_norm, activation_function, weight_decay):
    window_length = inputs.get_shape()[2]
    assert predictor_type in ['cnn', 'lstm'], 'type must be either cnn or lstm'
    if predictor_type == 'cnn':
        net = tflearn.conv_2d(inputs, 32, (1, 3), padding='valid', weights_init = 'xavier',\
                              regularizer='L2', weight_decay = weight_decay)
        if use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.activations.relu(net)
        net = activation_net(net, activation_function)

        net = tflearn.conv_2d(net, 32, (1, window_length - 2), padding='valid', weights_init = 'xavier',\
                              regularizer='L2', weight_decay = weight_decay)
        if use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.activations.relu(net)
        net = activation_net(net, activation_function)

        #################################################
        net = tflearn.conv_2d(net, 1, (1, 1), padding='valid', weights_init = 'xavier',\
                              regularizer='L2', weight_decay = weight_decay)
        if use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.activations.relu(net)
        net = activation_net(net, activation_function)
        ##################################################
        
        if DEBUG:
            print('After conv2d:', net.shape)
        net = tflearn.flatten(net)
        if DEBUG:
            print('Output:', net.shape)
    elif predictor_type == 'lstm':
        num_stocks = inputs.get_shape()[1]
        hidden_dim = 32
        net = tflearn.reshape(inputs, new_shape=[-1, window_length, feature_number])
        if DEBUG:
            print('Reshaped input:', net.shape)
        net = tflearn.lstm(net, hidden_dim, activation = activation_function,  weights_init = 'xavier')
        if DEBUG:
            print('After LSTM:', net.shape)
        net = tflearn.reshape(net, new_shape=[-1, num_stocks, hidden_dim,1]) ## reshape for conv2d in the next step
        if DEBUG:
            print('After reshape:', net.shape)

        #################################################
        net = tflearn.conv_2d(net, 1, (1, hidden_dim), padding='valid', weights_init = 'xavier',\
                              regularizer='L2', weight_decay = weight_decay)
        if use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.activations.relu(net)
        net = activation_net(net, activation_function)
        ##################################################
        net = tflearn.flatten(net)
        if DEBUG:
            print('Output:', net.shape)
    else:
        raise NotImplementedError
    return net


class StockActor(ActorNetwork):
    def __init__(self, sess, feature_number, state_dim, action_dim, action_bound, learning_rate,
                 decay_rate, decay_steps, weight_decay, tau, batch_size,
                 predictor_type, use_batch_norm,activation_function):
        self.feature_number = feature_number
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        self.activation_function = activation_function
        self.weight_decay = weight_decay
        ActorNetwork.__init__(self, sess, state_dim, action_dim, action_bound, learning_rate,
                              decay_rate, decay_steps, tau, batch_size)

    def create_actor_network(self):
        """
        self.s_dim: a list specifies shape, 
        s = [num_stock, window_len], action_dim[num_stock,1]
        """
        activation_function = self.activation_function
        nb_classes, window_length = self.s_dim
        assert nb_classes == self.a_dim[0]
        assert window_length > 2, 'This architecture only support window length larger than 2.'
        inputs = tflearn.input_data(shape=[None] + self.s_dim + [self.feature_number], name='input')
        net = stock_predictor(inputs, self.feature_number, self.predictor_type, \
                              self.use_batch_norm, self.activation_function, self.weight_decay)

        previous_w = tflearn.input_data(shape=[None] + self.a_dim, name = 'previous_w')
        net = tf.concat([net, previous_w], axis = 1)

        if DEBUG:
            print("after add previous_w ",net.shape)
        
        net = tflearn.fully_connected(net, 8 * self.a_dim[0], weights_init = 'xavier', regularizer='L2', weight_decay = self.weight_decay)
        if self.use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.activations.relu(net)
        net = activation_net(net, activation_function)
        
        net = tflearn.fully_connected(net, 8 * self.a_dim[0], weights_init = 'xavier', regularizer='L2', weight_decay = self.weight_decay)
        if self.use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.activations.relu(net)
        net = activation_net(net, activation_function)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, self.a_dim[0], activation='softmax',  weights_init = 'xavier', regularizer='L2', weight_decay = self.weight_decay)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out, previous_w

    def train(self, inputs, a_gradient, previous_w, global_step):
        '''
        input = [batch, num_stock, window, feature ]
        '''
        # self.global_step = self.global_step + 1
        # onestep = tf.constant(1)
        # self.global_step = onestep + self.global_step
        # print("actor global_step is ", self.sess.run(self.global_step))
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient,
            self.previous_w: previous_w,
            self.global_step: global_step
        })

    def predict(self, inputs, previous_w):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs,
            self.previous_w: previous_w
        })

    def predict_target(self, inputs, target_previous_w):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs,
            self.target_previous_w: target_previous_w
        })


class StockCritic(CriticNetwork):
    def __init__(self, sess, feature_number, state_dim, action_dim, learning_rate,
                 decay_rate, decay_steps, weight_decay, tau, num_actor_vars,
                 predictor_type, use_batch_norm, activation_function):
        self.feature_number = feature_number
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        self.activation_function = activation_function
        self.weight_decay = weight_decay
        CriticNetwork.__init__(self, sess, state_dim, action_dim, learning_rate,
                               decay_rate, decay_steps, tau, num_actor_vars)

    def create_critic_network(self):
        activation_function = self.activation_function
        inputs = tflearn.input_data(shape=[None] + self.s_dim + [self.feature_number])
        action = tflearn.input_data(shape=[None] + self.a_dim)
        net = stock_predictor(inputs, self.feature_number, self.predictor_type,\
                              self.use_batch_norm, self.activation_function, self.weight_decay)

        previous_w = tflearn.input_data(shape=[None] + self.a_dim, name = 'previous_w')
        net = tf.concat([net, previous_w], axis = 1)

        if DEBUG:
            print("after add previous_w ",net.shape)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 8 * self.a_dim[0], regularizer='L2', weight_decay = self.weight_decay)
        t2 = tflearn.fully_connected(action, 8 * self.a_dim[0], regularizer='L2', weight_decay = self.weight_decay)

        net = tf.add(t1, t2)
        if self.use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.activations.relu(net)
        net = activation_net(net, activation_function)


        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init, regularizer='L2', \
                                      weight_decay = self.weight_decay)
        return inputs, action, out, previous_w

    def train(self, inputs, action, predicted_q_value, previous_w, global_step):
        # self.global_step = self.global_step + 1
        # onestep = tf.constant(1)
        # self.global_step = onestep + self.global_step
        # print("critic global_step is ", self.sess.run(self.global_step))
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value,
            self.previous_w: previous_w,
            self.global_step: global_step
        })

    def predict(self, inputs, action, previous_w):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.previous_w: previous_w
        })

    def predict_target(self, inputs, action, target_previous_w):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action,
            self.target_previous_w: target_previous_w
        })

    def action_gradients(self, inputs, actions, previous_w):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions,
            self.previous_w: previous_w
        })


def obs_normalizer(observation, feature_number):
    """ Preprocess observation obtained by environment

    Args:
        observation: (nb_classes, window_length, num_features) or with info

    Returns: normalized

    """
 
    if isinstance(observation, tuple):
        observation = observation[0]

    if feature_number == 1: # directly use close/open ratio as feature
        observation = observation[:, :, 3:4] / observation[:, :, 0:1]
        observation = normalize(observation)

    if feature_number == 2: # use close/open ratio and volumn / volumn_last ratio
        observation_price = observation[:,:, 3:4] / observation[:,:, 0:1]
        observation_price = normalize(observation_price)
        observation_volume = stats.zscore(observation[:,:,5:6].astype('float64'), axis = 1)
        observation_volume = np.nan_to_num(observation_volume)
        observation = np.concatenate((observation_price, observation_volume), axis = 2)

    if feature_number == 4: # use open,low,high,close/close
        observation = observation[:,:, 0:4]/observation[:,-1:,3:4]
        observation = normalize(observation)

    if feature_number == 5: # use open,low,high,close/close and volumn / volumn_last ratio
        observation_price = observation[:,:, 0:4]/ observation[:,-1:,3:4]
        observation_price = normalize(observation_price)
        observation_volume = stats.zscore(observation[:,:,5:6].astype('float64'), axis = 1)
        observation_volume = np.nan_to_num(observation_volume)
        observation = np.concatenate((observation_price, observation_volume), axis = 2)

    return observation


def test_model(env, model):
    observation, info = env.reset()
    done = False
    while not done:
        action = model.predict_single(observation)
        observation, _, done, _ = env.step(action)
    env.render()


def test_model_multiple(env, models):
    observation, info = env.reset()
    done = False
    while not done:
        actions = []
        for model in models:
            actions.append(model.predict_single(observation))
        actions = np.array(actions)
        observation, _, done, info = env.step(actions)
    env.render()




class ddpg_trading_train(object):
    def __init__(self, config ,DEBUG = False):
        self.config = copy.deepcopy(config)
        self.device = config["device"]
        self.feature_number = config["input"]["feature_number"]
        self.window_length = config["input"]["window_length"]
        self.target_stocks = config["input"]["stocks"]
        self.nb_classes = len(self.target_stocks) + 1 # including cash account
        self.training_start_time = config["input"]["training_start_time"]
        self.training_end_time = config["input"]["training_end_time"]
        self.eval_start_time = config["input"]["eval_start_time"]
        self.eval_end_time = config["input"]["eval_end_time"]
        self.predictor_type = config["input"]["predictor_type"]
        self.use_batch_norm = config["input"]["use_batch_norm"]
        self.trading_cost = config["input"]["trading_cost"]
        self.time_cost = config["input"]["time_cost"]
        self.batch_size = config["training"]["batch size"]
        self.action_bound = config["training"]["action_bound"]
        self.tau = config["training"]["tau"]
        self.activation_function = config["layers"]["activation_function"]
        self.action_dim, self.state_dim = [self.nb_classes], [self.nb_classes, self.window_length]
        self.model_save_path, self.summary_path, self.train_id = generate_train_folder(config) 


    def train_model(self):
        print("training period is from %s to %s"%(self.training_start_time,self.training_end_time))
        
        self.target_history, _, self.training_start_time, self.training_end_time \
                                = fetch_data(start_time = self.training_start_time, 
                                             end_time = self.training_end_time, 
                                             window_length = self.window_length,
                                             stocks = self.target_stocks)
        
        print("total training example is %d" %(self.training_start_time-self.training_end_time))
        print("self.target_history shape is", self.target_history.shape)  
        if self.config["training"]["max_step"] <= 0:
            self.config["training"]["max_step"] = self.target_history.shape[1] - self.window_length-1
            # print("max_steps is", self.target_history.shape[1] - self.window_length-1)

        env = PortfolioEnv(history = self.target_history, 
                           abbreviation = self.target_stocks, 
                           steps = self.config["training"]["max_step"],
                           trading_cost = self.trading_cost,
                           time_cost = self.time_cost, 
                           window_length = self.window_length,
                           reward_function = self.config["input"]["reward_function"])

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim))
        
        variable_scope = get_variable_scope(self.window_length, self.predictor_type, self.use_batch_norm)
        if self.config["device"] == "cpu":
            device_res = "/cpu:0"
        else:
            device_res = "/gpu:0"
        print("device is ", device_res)
        with tf.device(device_res):
            self.sess = self.start_session()
            with tf.variable_scope(variable_scope):
                actor = StockActor(sess = self.sess, 
                                   feature_number = self.feature_number,
                                   state_dim = self.state_dim, 
                                   action_dim = self.action_dim, 
                                   action_bound = self.action_bound, 
                                   learning_rate = self.config["training"]["actor learning rate"], 
                                   decay_rate = self.config["training"]["actor decay rate"],
                                   decay_steps = self.config["training"]["actor decay steps"],
                                   weight_decay = self.config["training"]["actor weight decay"],
                                   tau = self.tau, 
                                   batch_size = self.batch_size,
                                   predictor_type = self.predictor_type, 
                                   use_batch_norm = self.use_batch_norm,
                                   activation_function = self.activation_function)
                critic = StockCritic(sess = self.sess, 
                                     feature_number = self.feature_number,
                                     state_dim = self.state_dim, 
                                     action_dim = self.action_dim, 
                                     tau = self.tau,
                                     learning_rate = self.config["training"]["critic learning rate"], 
                                     decay_rate = self.config["training"]["critic decay rate"],
                                     decay_steps = self.config["training"]["critic decay steps"],
                                     weight_decay = self.config["training"]["critic weight decay"],
                                     num_actor_vars = actor.get_num_trainable_vars(),
                                     predictor_type = self.predictor_type, 
                                     use_batch_norm = self.use_batch_norm,
                                     activation_function = self.activation_function)
                ddpg_model = DDPG(env = env,
                                  sess = self.sess,
                                  actor = actor, 
                                  critic = critic, 
                                  actor_noise = actor_noise, 
                                  obs_normalizer = obs_normalizer,
                                  config = self.config, 
                                  model_save_path = self.model_save_path,
                                  summary_path = self.summary_path)
                ddpg_model.initialize(load_weights = False)
                ddpg_model.train()
                self.close_session()
        return self.train_id

    def start_session(self):
        tf.reset_default_graph()
        tf_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False,)
        # if self.device == "cpu":
        #     tf_config.gpu_options.per_process_gpu_memory_fraction = 0
        # else:
        #     tf_config.gpu_options.per_process_gpu_memory_fraction = 1
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config = tf_config)
        tflearn.config.init_training_mode()
        return sess

    def close_session(self):
        self.sess.close()

if __name__ == '__main__':

    parser = build_parser()
    args = vars(parser.parse_args())
    pprint.pprint(args)

    if args['debug'] == 'True':
        DEBUG = True
    else:
        DEBUG = False

    model = ddpg_trading_train(load_config(), DEBUG = DEBUG)
    train_id = model.train_model()
    print(train_id)
