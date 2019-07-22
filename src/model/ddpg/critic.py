"""
Critic Network definition, the input is (o, a_{t-1}, a_t) since (o, a_{t-1}) is the state.
Basically, it evaluates the value of (current action, previous action and observation) pair
"""

import tensorflow as tf
import tflearn


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, decay_rate, decay_steps, 
                tau, num_actor_vars):
        self.sess = sess
        assert isinstance(state_dim, list), 'state_dim must be a list.'
        self.s_dim = state_dim
        assert isinstance(action_dim, list), 'action_dim must be a list.'
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.tau = tau

        # Create the critic network
        self.inputs, self.action, self.out, self.previous_w = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out, self.target_previous_w = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # set decay learning
        self.global_step = tf.Variable(0, trainable = False)
        self.decay_learning_rate = tf.train.exponential_decay(learning_rate = self.learning_rate, 
                                                         global_step = self.global_step, 
                                                         decay_steps = self.decay_steps,
                                                         decay_rate = self.decay_rate, 
                                                         staircase = True)
        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.decay_learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        raise NotImplementedError('Create critic should return (inputs, action, out)')

    # def train(self, inputs, action, predicted_q_value):
    #     return self.sess.run([self.out, self.optimize], feed_dict={
    #         self.inputs: inputs,
    #         self.action: action,
    #         self.predicted_q_value: predicted_q_value
    #     })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def show_learning_rate(self, global_step):
        print("decay global_step is", self.sess.run(self.global_step, feed_dict={
                                                    self.global_step: global_step        
            }))
        print("decay_learning_rate is", self.sess.run(self.decay_learning_rate, feed_dict={
                                                      self.global_step: global_step
            }))
        return
