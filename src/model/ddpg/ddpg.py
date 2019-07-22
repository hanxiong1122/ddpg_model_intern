"""
The deep deterministic policy gradient model. Contains main training loop and deployment
"""
from __future__ import print_function

import os
import traceback
import json
import numpy as np
import tensorflow as tf
import time
import math

from utils.data import read_stock_history, normalize, read_uptodate_data, fetch_data
from .replay_buffer import ReplayBuffer
from ..base_model import BaseModel
from utils.clientutils import *





class DDPG(BaseModel):
    def __init__(self, env, sess, actor, critic, actor_noise, 
                 obs_normalizer=None, 
                 action_processor=None,
                 config = None,
                 model_save_path='weights/ddpg/ddpg.ckpt', summary_path='results/ddpg/'):
        # with open(config_file) as f:
        #     self.config = json.load(f)
        self.config = config
        assert self.config != None, "Can't load config file"
        np.random.seed(self.config["training"]['seed'])
        if env:
            env.seed(self.config["training"]['seed'])
        self.model_save_path = model_save_path
        self.summary_path = summary_path
        self.sess = sess
        # if env is None, then DDPG just predicts
        self.env = env
        self.actor = actor
        self.critic = critic
        self.actor_noise = actor_noise
        self.feature_number = self.config["input"]["feature_number"]
        self.window_length = self.config["input"]["window_length"]
        self.obs_normalizer = obs_normalizer
        self.action_processor = action_processor
        self.nb_classes = len(self.config["input"]["stocks"]) + 1
        self.device = config["device"]
        # if self.device == "cpu":
        #     device_res = "/cpu:0"
        # else:
        #     device_res = "/gpu:0"
        #     print("Initialize summary_ops and loss_summary_ops using kernel ", device_res)
        # with tf.device(device_res):
        #     self.summary_ops, self.summary_vars = self.build_summaries()
        #     self.loss_summary_ops, self.loss_summary_vars = self.build_loss_summary()
        self.summary_ops, self.summary_vars = self.build_summaries()
        self.loss_summary_ops, self.loss_summary_vars = self.build_loss_summary()
        self.min_gradient_ops, self.min_gradient_vars = self.build_critic_actor_gradient()

    def initialize(self, load_weights=True, verbose=True):
        """ Load training history from path. To be add feature to just load weights, not training states

        """
        if self.device == "cpu":
            device_res = "/cpu:0"
        else:
            device_res = "/gpu:0"
        print("Initialize network using kernel ", device_res)
        with tf.device(device_res):            
            if load_weights:
                try:
                    variables = tf.global_variables()
                    param_dict = {}
                    saver = tf.train.Saver()
                    saver.restore(self.sess, self.model_save_path)
                    for var in variables:
                        var_name = var.name[:-2]
                        if verbose:
                            print('Loading {} from checkpoint. Name: {}'.format(var.name, var_name))
                        param_dict[var_name] = var
                    print("model is loaded")
                except:
                    traceback.print_exc()
                    print('Build model from scratch')
                    self.sess.run(tf.global_variables_initializer())
            else:
                print('Build model from scratch')
                self.sess.run(tf.global_variables_initializer())

    def train(self, save_every_episode=1, verbose=True, debug=False):
        """ Must already call intialize

        Args:
            save_every_episode:
            print_every_step:
            verbose:
            debug:

        Returns:

        """
        writer = tf.summary.FileWriter(self.summary_path, self.sess.graph)

        self.actor.update_target_network()
        self.critic.update_target_network()

        np.random.seed(self.config["training"]['seed'])
        num_episode = self.config["training"]['episode']
        batch_size = self.config["training"]['batch size']
        gamma = self.config["training"]['gamma']
        self.buffer = ReplayBuffer(self.config["training"]['buffer size'])

        self.eval_start_time = self.config["input"]["eval_start_time"]
        self.eval_end_time = self.config["input"]["eval_end_time"]
        print("Date of evaluate is from %s to %s"%(self.eval_start_time,self.eval_end_time))        
        self.eval_history, _, self.eval_start_time, self.eval_end_time \
                                = fetch_data(start_time = self.eval_start_time, 
                                             end_time = self.eval_end_time, 
                                             window_length = self.window_length,
                                             stocks = self.config["input"]["stocks"])
        print("total eval example is %d" %(self.eval_start_time-self.eval_end_time))

        # main training loop
        # if self.device == "cpu":
        #     device_res = "/cpu:0"
        # else:
        #     device_res = "/gpu:0"
        # print("device is ", device_res)
        # with tf.device(device_res):
        training_start_time = time.time()
        print("The training max step in each episode is ", self.config["training"]["max_step"])
        max_reward = -math.inf
        best_episode = 0  
        global_step = 0
        for i in range(num_episode):
            episode_start_time = time.time()
            if verbose and debug:
                print("Episode: " + str(i) + " Replay Buffer " + str(self.buffer.count()))

            previous_observation = self.env.reset()

            previous_w = np.zeros(shape=(self.nb_classes))
            previous_w[0] = 1


            if self.obs_normalizer:
                previous_observation = self.obs_normalizer(previous_observation, self.feature_number)
            ep_reward = 0
            ep_ave_max_q = 0
            # keeps sampling until done
            for j in range(self.config["training"]['max_step']):            
                action = self.actor.predict(np.expand_dims(previous_observation, axis=0), \
                                            previous_w.reshape(-1, self.nb_classes)).squeeze(
                                            axis=0) + self.actor_noise()

                if self.action_processor:
                    action_take = self.action_processor(action)
                else:
                    action_take = action
                # step forward
                observation, reward, done, _ = self.env.step(action_take)

                if self.obs_normalizer:
                    observation = self.obs_normalizer(observation, self.feature_number)

                # add to buffer
                self.buffer.add(previous_observation, action, reward, done, observation, previous_w)

                if self.buffer.size() >= batch_size and j%200 == 0:
                    for t in range(50):
                        # batch update
                        s_batch, a_batch, r_batch, t_batch, s2_batch, previous_w_batch \
                                             = self.buffer.sample_batch(batch_size)

                        target_q = self.critic.predict_target(s2_batch, self.actor.predict_target(s2_batch, a_batch), a_batch)
                        
                        y_i = []
                        for k in range(batch_size):
                            if t_batch[k]:
                                y_i.append(r_batch[k])
                            else:
                                y_i.append(r_batch[k] + gamma * target_q[k])

                        # Update the critic given the targets
                        predicted_q_value, _ = self.critic.train(
                            s_batch, a_batch, np.reshape(y_i, (batch_size, 1)), previous_w_batch, global_step)

                        ep_ave_max_q += np.amax(predicted_q_value)

                        # Update the actor policy using the sampled gradient
                        a_outs = self.actor.predict(s_batch, previous_w_batch)
                        grads = self.critic.action_gradients(s_batch, a_outs, previous_w_batch)
                        # print(np.amin(grads))
                        self.actor.train(s_batch, grads[0], previous_w_batch, global_step)

                        # Update target networks
                        self.actor.update_target_network()
                        self.critic.update_target_network()

                        if t == 0:
                            loss_q_value = tf.reduce_sum(tf.square(y_i - predicted_q_value)) / batch_size
                            loss_q = self.sess.run(loss_q_value)
                            loss_summary_str = self.sess.run(self.loss_summary_ops, feed_dict={
                                self.loss_summary_vars[0]: loss_q
                                })
                            writer.add_summary(loss_summary_str, global_step)

                            min_gradient_str = self.sess.run(self.min_gradient_ops, feed_dict={
                                self.min_gradient_vars[0]: np.amax(np.absolute(grads))
                                })
                            writer.add_summary(min_gradient_str, global_step)
                            writer.flush()

                        # Update global step
                        global_step += 1

                ep_reward += reward
                previous_observation = observation
                previous_w = action

                if done or j == self.config["training"]['max_step'] - 1:
                    summary_str = self.sess.run(self.summary_ops, feed_dict={
                        self.summary_vars[0]: ep_reward,
                        self.summary_vars[1]: ep_ave_max_q / float(j)
                    })

                    writer.add_summary(summary_str, i)
                    writer.flush()

                    print(Stage.TRAINING)

                    print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}'.format(i, ep_reward, (ep_ave_max_q / float(j))))

                    percent = ((i + 1) / num_episode) * 100
                    report_model_progress(self.config["model_id"],info={"stage": Stage.TRAINING, "step": i + 1, "percent": percent, "pv": ep_reward})


                    break
            episode_end_time = time.time()
            print("This episode running time is ", episode_end_time - episode_start_time)
            # if i == 0 or (max_reward <= ep_reward):
            #     max_reward = ep_reward
            #     best_episode = i
            #     save_model_start_time = time.time()
            #     self.save_model(verbose=True)
            #     save_model_end_time = time.time()
            #     print("It takes %f to save model" %(save_model_end_time - save_model_start_time))
            # print("The best model happens in the %d step, the best reward is %f" %(best_episode, max_reward))

            # show learning rate
            # t1=time.time()
            self.actor.show_learning_rate(global_step)
            self.critic.show_learning_rate(global_step)
            # t2=time.time()
            # print("t2-t1 ", t2-t1)

            eval_value = self.eval_model()
            if i == 0 or (max_reward <= eval_value):
                max_reward = eval_value
                best_episode = i
                # save_model_start_time = time.time()
                # self.save_model(verbose=True)
                # save_model_end_time = time.time()
                # print("It takes %f to save model" %(save_model_end_time - save_model_start_time))
            print("The eval_value in this episode is %f" %(eval_value))
            print("The best model happens in the %d step, the best eval_value is %f" %(best_episode, max_reward))

        self.save_model(verbose=True)
        training_end_time = time.time()
        print("This training time is ", training_end_time - training_start_time)
        print('Finish.')

    def predict(self, observation, previous_w, feature_number):
        """ predict the next action using actor model, only used in deploy.
            Can be used in multiple environments.

        Args:
            observation: (batch_size, num_stocks + 1, window_length)

        Returns: action array with shape (batch_size, num_stocks + 1)

        """
        if self.obs_normalizer:
            observation = self.obs_normalizer(observation, feature_number)
        action = self.actor.predict(observation, previous_w.reshape(-1, self.nb_classes))
        if self.action_processor:
            action = self.action_processor(action)
        return action

    def predict_single(self, observation, previous_w, feature_number):
        """ Predict the action of a single observation

        Args:
            observation: (num_stocks + 1, window_length)

        Returns: a single action array with shape (num_stocks + 1,)

        """
        if self.obs_normalizer:
            observation = self.obs_normalizer(observation, feature_number)
        action = self.actor.predict(np.expand_dims(observation, axis=0), previous_w.reshape(-1, self.nb_classes)).squeeze(axis=0)
        if self.action_processor:
            action = self.action_processor(action)
        return action

    def eval_model(self):
        print("eval period is from %s to %s"\
            %(self.config["input"]["eval_start_time"],self.config["input"]["eval_end_time"]))
        nb_classes = len(self.config["input"]["stocks"]) + 1

        prev_pv = 1
        prev_action = np.zeros(shape=(1,nb_classes))
        prev_action[0][0] = 1

        for step in range(-self.eval_end_time +self.eval_start_time):
            observation = self.eval_history[:, step:step + self.window_length, :].copy()
            cash_observation = np.ones((1, self.window_length, observation.shape[2]))
            observation = np.concatenate((cash_observation, observation), axis=0)
            action = self.predict_single(observation, prev_action, self.feature_number)

            date = step + self.window_length
            today_price = self.eval_history[:, date-1, 4]
            prev_price = self.eval_history[:, date-2, 4]
            cash_price = np.ones((1,))
            today_price = np.concatenate((cash_price, today_price), axis=0)
            prev_price = np.concatenate((cash_price, prev_price), axis=0)
            y_t = (today_price/prev_price)
    
            unadjust_weight = (y_t.flatten() * prev_action.flatten()) / np.dot(y_t.flatten(), prev_action.flatten())
            unadjust_pv = prev_pv * np.dot(y_t.flatten(), prev_action.flatten())
            adjust_weight = action
            mu1 = self.config["input"]["trading_cost"] * (np.abs(unadjust_weight - adjust_weight)).sum()
            adjust_pv = (1-mu1) * unadjust_pv

            # update action and pv
            prev_action = action
            prev_pv = adjust_pv
        return prev_pv


    def save_model(self, verbose=False):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)

        saver = tf.train.Saver()
        model_path = saver.save(self.sess, self.model_save_path)
        print("Model saved in %s" % model_path)


    def build_loss_summary(self):
        loss_q = tf.Variable(0, dtype = tf.float64)
        with tf.device("/cpu:0"):
            loss_q_summary = tf.summary.scalar("loss_q", loss_q)
            loss_summary_vars = [loss_q]
            loss_summary_ops = tf.summary.merge([loss_q_summary])
        return loss_summary_ops, loss_summary_vars


    def build_summaries(self):
        episode_reward = tf.Variable(0, dtype = tf.float64)
        with tf.device("/cpu:0"):
            episode_reward_summary = tf.summary.scalar("Reward", episode_reward)
            episode_ave_max_q = tf.Variable(0, dtype = tf.float64)
            episode_ave_max_q_summary = tf.summary.scalar("Qmax_Value", episode_ave_max_q)

            summary_vars = [episode_reward, episode_ave_max_q]
            summary_ops = tf.summary.merge([episode_reward_summary,episode_ave_max_q_summary])
        return summary_ops, summary_vars

    def build_critic_actor_gradient(self):
        min_gradient = tf.Variable(0, dtype = tf.float64)
        with tf.device("/cpu:0"):
            min_gradient_summary = tf.summary.scalar("critic_network_gradient", min_gradient)
            min_gradient_vars = [min_gradient]
            min_gradient_ops = tf.summary.merge([min_gradient_summary])
        return min_gradient_ops, min_gradient_vars


