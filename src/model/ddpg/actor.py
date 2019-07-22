"""
Actor Network definition, The CNN architecture follows the one in this paper
https://arxiv.org/abs/1706.10059
Author: Patrick Emami, Modified by Chi Zhang
"""

import tensorflow as tf


# ===========================
#   Actor DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, decay_rate,
                decay_steps, tau, batch_size):
        """

        Args:
            sess: a tensorflow session
            state_dim: a list specifies shape
            action_dim: a list specified action shape
            action_bound: whether to normalize action in the end
            learning_rate: learning rate
            tau: target network update parameter
            batch_size: use for normalization
        """
        self.sess = sess
        assert isinstance(state_dim, list), 'state_dim must be a list.'
        self.s_dim = state_dim
        assert isinstance(action_dim, list), 'action_dim must be a list.'
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out, self.previous_w = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out, self.target_previous_w = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None] + self.a_dim)

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # set decay learning
        self.global_step = tf.Variable(0, trainable = False)
        self.decay_learning_rate = tf.train.exponential_decay(learning_rate = self.learning_rate, 
                                                         global_step = self.global_step, 
                                                         decay_steps = self.decay_steps,
                                                         decay_rate = self.decay_rate, 
                                                         staircase = True)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.decay_learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        raise NotImplementedError('Create actor should return (inputs, out, scaled_out)')

    # def train(self, inputs, a_gradient):
    #     self.sess.run(self.optimize, feed_dict={
    #         self.inputs: inputs,
    #         self.action_gradient: a_gradient
    #     })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def show_learning_rate(self,global_step):
        print("actor global_step is", self.sess.run(self.global_step, feed_dict={
                                                    self.global_step: global_step
            }))
        print("actor learning_rate is ", self.sess.run(self.decay_learning_rate, feed_dict={
                                                       self.global_step: global_step
            }))
        return