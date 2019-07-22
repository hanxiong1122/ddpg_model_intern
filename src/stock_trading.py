"""
Use DDPG to train a stock trader based on a window of history price
"""

from __future__ import print_function, division

from model.ddpg.actor import ActorNetwork
from model.ddpg.critic import CriticNetwork
from model.ddpg.ddpg import DDPG
from model.ddpg.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

from environment.portfolio import PortfolioEnv
from utils.data import read_stock_history, normalize
from utils.modelutils import  generate_train_folder, load_config, build_parser


import os
import numpy as np
import tflearn
import tensorflow as tf
import argparse
import pprint
import logging
import json


DEBUG = False



def get_variable_scope(window_length, predictor_type, use_batch_norm):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'
    return '{}_window_{}_{}'.format(predictor_type, window_length, batch_norm_str)


def stock_predictor(inputs, predictor_type, use_batch_norm):
    window_length = inputs.get_shape()[2]
    assert predictor_type in ['cnn', 'lstm'], 'type must be either cnn or lstm'
    if predictor_type == 'cnn':
        net = tflearn.conv_2d(inputs, 32, (1, 3), padding='valid')
        if use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.conv_2d(net, 32, (1, window_length - 2), padding='valid')
        if use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        if DEBUG:
            print('After conv2d:', net.shape)
        net = tflearn.flatten(net)
        if DEBUG:
            print('Output:', net.shape)
    elif predictor_type == 'lstm':
        num_stocks = inputs.get_shape()[1]
        hidden_dim = 32
        net = tflearn.reshape(inputs, new_shape=[-1, window_length, 1])
        if DEBUG:
            print('Reshaped input:', net.shape)
        net = tflearn.lstm(net, hidden_dim)
        if DEBUG:
            print('After LSTM:', net.shape)
        net = tflearn.reshape(net, new_shape=[-1, num_stocks, hidden_dim])
        if DEBUG:
            print('After reshape:', net.shape)
        net = tflearn.flatten(net)
        if DEBUG:
            print('Output:', net.shape)
    else:
        raise NotImplementedError

    return net


class StockActor(ActorNetwork):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size,
                 predictor_type, use_batch_norm):
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        ActorNetwork.__init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size)

    def create_actor_network(self):
        """
        self.s_dim: a list specifies shape, 
        s = [num_stock, window_len], action_dim[num_stock,1]
        """
        nb_classes, window_length = self.s_dim
        assert nb_classes == self.a_dim[0]
        assert window_length > 2, 'This architecture only support window length larger than 2.'
        inputs = tflearn.input_data(shape=[None] + self.s_dim + [1], name='input')

        net = stock_predictor(inputs, self.predictor_type, self.use_batch_norm)

        net = tflearn.fully_connected(net, 64)
        if self.use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 64)
        if self.use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, self.a_dim[0], activation='softmax', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        '''
        input = [batch, num_stock, window, feature ]
        '''
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })


class StockCritic(CriticNetwork):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars,
                 predictor_type, use_batch_norm):
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        CriticNetwork.__init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None] + self.s_dim + [1])
        action = tflearn.input_data(shape=[None] + self.a_dim)

        net = stock_predictor(inputs, self.predictor_type, self.use_batch_norm)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 64)
        t2 = tflearn.fully_connected(action, 64)

        net = tf.add(t1, t2)
        if self.use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })


def obs_normalizer(observation):
    """ Preprocess observation obtained by environment

    Args:
        observation: (nb_classes, window_length, num_features) or with info

    Returns: normalized

    """
    if isinstance(observation, tuple):
        observation = observation[0]
    # directly use close/open ratio as feature
    observation = observation[:, :, 3:4] / observation[:, :, 0:1]
    observation = normalize(observation)
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









if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Provide arguments for training different DDPG models')

    # parser.add_argument('--debug', '-d', help='print debug statement', default=False)
    # parser.add_argument('--predictor_type', '-p', help='cnn or lstm predictor', required=True)
    # parser.add_argument('--window_length', '-w', help='observation window length', required=True)
    # parser.add_argument('--batch_norm', '-b', help='whether to use batch normalization', required=True)
    ddpg_logger = logging.getLogger("DDPG")
    ddpg_logger.setLevel(logging.DEBUG)

    parser = build_parser()
    args = vars(parser.parse_args())
    pprint.pprint(args)

    if args['debug'] == 'True':
        DEBUG = True
    else:
        DEBUG = False

    config = load_config()
    window_length = config["input"]["window_length"]
    target_stocks = config["input"]["stocks"]
    nb_classes = len(target_stocks) + 1 # including cash account
    num_training_time = config["input"]["num_training_time"]
    predictor_type = config["input"]["predictor_type"]
    use_batch_norm = config["input"]["use_batch_norm"]
    trading_cost = config["input"]["trading_cost"]
    time_cost = config["input"]["time_cost"]
    batch_size = config["training"]["batch size"]
    action_bound = config["training"]["action_bound"]
    tau = config["training"]["tau"]
    max_step = config["training"]["max_step"]
    action_dim, state_dim = [nb_classes], [nb_classes, window_length]



    model_save_path, summary_path, train_id = generate_train_folder(config)

    # fetch data
    history, abbreviation = read_stock_history(filepath='utils/datasets/stocks_history_target.h5')
    #history [stock, time, feature] does not include volume
    history = history[:, :, :4]

    # get target history
    target_history = np.empty(shape=(len(target_stocks), num_training_time, history.shape[2]))
    for i, stock in enumerate(target_stocks):
        target_history[i] = history[abbreviation.index(stock), :num_training_time, :]

    # setup environment // starting state is hardcode
    env = PortfolioEnv(history = target_history, 
                       abbreviation = target_stocks, 
                       steps = max_step,
                       trading_cost = trading_cost,
                       time_cost = time_cost, 
                       window_length = window_length)
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))


    variable_scope = get_variable_scope(window_length, predictor_type, use_batch_norm)

    with tf.variable_scope(variable_scope):
        sess = tf.Session()
        actor = StockActor(sess = sess, 
                           state_dim = state_dim, 
                           action_dim = action_dim, 
                           action_bound = action_bound, 
                           learning_rate = config["training"]["actor learning rate"], 
                           tau = tau, 
                           batch_size = batch_size,
                           predictor_type = predictor_type, 
                           use_batch_norm = use_batch_norm)
        critic = StockCritic(sess=sess, 
                             state_dim=state_dim, 
                             action_dim=action_dim, 
                             tau=tau,
                             learning_rate=config["training"]["critic learning rate"], 
                             num_actor_vars=actor.get_num_trainable_vars(),
                             predictor_type=predictor_type, 
                             use_batch_norm=use_batch_norm)
        ddpg_model = DDPG(env = env, 
                          sess = sess, 
                          actor = actor, 
                          critic = critic, 
                          actor_noise = actor_noise, 
                          obs_normalizer=obs_normalizer,
                          config = config, 
                          model_save_path=model_save_path,
                          summary_path=summary_path)
        ddpg_model.initialize(load_weights=False)
        ddpg_model.train()
