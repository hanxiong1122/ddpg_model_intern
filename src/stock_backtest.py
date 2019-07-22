from model.ddpg.actor import ActorNetwork
from model.ddpg.critic import CriticNetwork
from model.ddpg.ddpg import DDPG
from model.ddpg.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

import numpy as np
import tflearn
import tensorflow as tf
import json

from stock_model_train import StockActor, StockCritic, obs_normalizer,\
                          test_model, get_variable_scope, test_model_multiple, obs_normalizer

from utils.clientutils import *

    
from model.supervised.lstm import StockLSTM
from model.supervised.cnn import StockCNN
from utils.data import read_stock_history, read_uptodate_data, fetch_data
from utils.modelutils import *
import copy



class ddpg_restore_model():
    def __init__(self, train_id):
        self.model_save_path = get_model_path(train_id)
        self.summary_path = get_result_path(train_id)
        self.config_path = "./train_package/" + str(train_id) + "/stock.json"
        self.config = load_config(config_path = self.config_path)
        self.batch_size = self.config["training"]["batch size"]
        self.action_bound = self.config["training"]["action_bound"]
        self.tau = self.config["training"]["tau"]
        self.feature_number = self.config["input"]["feature_number"]
        self.window_length = self.config["input"]["window_length"]
        self.predictor_type = self.config["input"]["predictor_type"]
        self.use_batch_norm = self.config["input"]["use_batch_norm"]
        self.testing_stocks = self.config["input"]["stocks"]
        self.trading_cost = self.config["input"]["trading_cost"]
        self.time_cost = self.config["input"]["time_cost"]
        self.testing_start_time = self.config["testing"]["testing_start_time"]
        self.testing_end_time = self.config["testing"]["testing_end_time"]
        self.activation_function = self.config["layers"]["activation_function"]
        self.train_id  = train_id

    def restore(self):
        self.start_session()
        nb_classes = len(self.testing_stocks) + 1
        action_dim, state_dim = [nb_classes], [nb_classes, self.window_length]
        variable_scope = get_variable_scope(self.window_length, self.predictor_type, self.use_batch_norm)
        with tf.variable_scope(variable_scope):
            actor = StockActor(sess = self.sess,
                               feature_number = self.feature_number, 
                               state_dim = state_dim, 
                               action_dim = action_dim, 
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
                                 state_dim = state_dim, 
                                 action_dim = action_dim, 
                                 tau = self.tau,
                                 learning_rate = self.config["training"]["critic learning rate"], 
                                 decay_rate = self.config["training"]["critic decay rate"],
                                 decay_steps = self.config["training"]["critic decay steps"],
                                 weight_decay = self.config["training"]["critic weight decay"],
                                 num_actor_vars = actor.get_num_trainable_vars(), 
                                 predictor_type = self.predictor_type, 
                                 use_batch_norm = self.use_batch_norm,
                                 activation_function = self.activation_function)

            actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
            print(self.model_save_path)

            ddpg_model = DDPG(env = None, 
                              sess = self.sess, 
                              actor = actor, 
                              critic = critic, 
                              actor_noise = actor_noise, 
                              obs_normalizer = obs_normalizer,
                              config =self.config, 
                              model_save_path = self.model_save_path,
                              summary_path = self.summary_path)
            ddpg_model.initialize(load_weights=True, verbose=False)
            self.model = ddpg_model

    def predict_single(self, model, observation, previous_w):
        return model.predict_single(observation, previous_w, self.feature_number)


    # def fetch_data(self, testing_start_time, testing_end_time, window_length, testing_stocks):
    #     history, time_list = read_uptodate_data(testing_stocks, rootpath = "utils")
    #     total_length = len(time_list)
    #     try:
    #         if testing_start_time in time_list:
    #             self.testing_start_time = total_length - time_list.index(testing_start_time)
    #         else:
    #             for time in time_list:
    #                 if testing_start_time < time:
    #                     self.testing_start_time = total_length - time_list.index(time)
    #                     break;
    #         assert isinstance(self.testing_start_time, int), "not a valid testing start date"
    #     except ValueError as e:
    #         print('testing_start_time error', e)
    #     try:
    #         if testing_end_time in time_list:
    #             self.testing_end_time = total_length - time_list.index(testing_end_time)
    #         else:
    #             for i,time in enumerate(time_list):
    #                 if (i != len(time_list) - 1 and testing_end_time > time and testing_end_time < time_list[i+1]) or \
    #                     i == len(time_list) - 1:
    #                         self.testing_end_time = total_length - time_list.index(time)
    #                         break;
    #         assert isinstance(self.testing_start_time, int), "not a valid testing end date"
    #     except ValueError as e:
    #         print('testing_end_time error', e)
    #     # print(self.training_end_time, self.training_start_time)
    #     assert self.testing_end_time < self.testing_start_time, "testing start date must be earlier then traing end date"
    #     if self.testing_end_time == 1:
    #         print("The testing start date and end date1 are from %s to %s" %(time_list[-self.testing_start_time],\
    #                                                                         time_list[-1]))
    #         return history[:,-self.testing_start_time - window_length:,:],\
    #                 time_list[-self.testing_start_time:]
    #     else:
    #         print("The testing start date and end date2 are from %s to %s" %(time_list[-self.testing_start_time],\
    #                                                                         time_list[-self.testing_end_time]))
    #         return history[:,-self.testing_start_time - window_length: -self.testing_end_time,:], \
    #                 time_list[-self.testing_start_time: -self.testing_end_time]

    def backtest(self, start_date=None, end_date=None):
        if start_date is not None:
            self.testing_start_time = start_date
        if end_date is not None:
            self.testing_end_time = end_date

        print("bactesting period is from %s to %s"%(self.testing_start_time,self.testing_end_time))
        nb_classes = len(self.testing_stocks) + 1
        testing_history, self.time_list, self.testing_start_time, self.testing_end_time \
                             = fetch_data(start_time = self.testing_start_time,
                                          end_time = self.testing_end_time,
                                          window_length = self.window_length,
                                          stocks = self.testing_stocks)

        # print("this is testing_history")
        # print(testing_history)

        # print("teststage time list is ", time_list)
        # print(self.testing_start_time, self.testing_end_time)
        print("total testing example is %d" %(self.testing_start_time-self.testing_end_time))

        prev_pv = 1
        prev_action = np.zeros(shape=(1,nb_classes))
        prev_action[0][0] = 1
        action_list = []
        share_list, share_change_list = [], []
        pv_list, actualPV_list = [], []
        price_list =[]
        mu1_list = []

        #add a W' change list so we could see the change in weight every day 
        unadjust_weight_list = []

        for step in range(-self.testing_end_time +self.testing_start_time):
            observation = testing_history[:, step:step + self.window_length, :].copy()
            cash_observation = np.ones((1, self.window_length, observation.shape[2]))
            observation = np.concatenate((cash_observation, observation), axis=0)
            
            action = self.predict_single(self.model, observation, prev_action)

            date = step + self.window_length
            today_price = testing_history[:, date-1, 4]
            prev_price = testing_history[:, date-2, 4]
            cash_price = np.ones((1,))
            today_price = np.concatenate((cash_price, today_price), axis=0)
            prev_price = np.concatenate((cash_price, prev_price), axis=0)
            y_t = (today_price/prev_price)
            

            #unadjust_weight is the W', the weight change according to the price change 
            unadjust_weight = (y_t.flatten() * prev_action.flatten()) / np.dot(y_t.flatten(), prev_action.flatten())
            unadjust_pv = prev_pv * np.dot(y_t.flatten(), prev_action.flatten())
            adjust_weight = action
            mu1 = self.trading_cost * (np.abs(unadjust_weight - adjust_weight)).sum()
            adjust_pv = (1-mu1) * unadjust_pv
            adjust_pv = np.asarray(adjust_pv)
            unadjust_pv = np.asarray(unadjust_pv)
    
            share_change = (adjust_pv.flatten() * adjust_weight.flatten() - \
                            unadjust_pv.flatten() * unadjust_weight.flatten())/today_price

            share_change = share_change  # assume portfolio start from 1 million
            stock_return = (today_price.flatten()[1:] / today_price.flatten()[1:] - 1)if step == 0 \
                           else (today_price.flatten()[1:]/price_list[0][1:] - 1)


            pv_list.append(unadjust_pv)
            share_change_list.append(share_change)
            action_list.append(action)
            price_list.append(today_price.flatten())
            mu1_list.append(mu1)
            unadjust_weight_list.append(unadjust_weight)

            # update action and pv
            prev_action = action
            prev_pv = adjust_pv

        self.collection_testing_result(action_list, share_change_list, pv_list, price_list, testing_history, unadjust_weight_list)
        summary = self.save_to_file()

        if need_save_to_db():
            test_result = self.save_to_db(summary)

        if need_report_progress():
            report_model_progress(self.config["model_id"], info = {"stage": Stage.DONE, "test_result":test_result})

        self.close_session()

        return summary

    def collection_testing_result(self, action_list, share_change_list, pv_list, price_list, testing_history, unadjust_weight_list):
        self.action_list = np.asarray(action_list)
        self.unadjust_weight_list = np.asarray(unadjust_weight_list)
        self.share_change_list = np.asarray(share_change_list)
        self.pv_list = np.asarray(pv_list)

        share_list = np.asarray(share_change_list)
        share_list[0][0] +=1
        share_list = np.cumsum(np.asarray(share_list), axis = 0) 
        self.share_list = share_list 

        close_price = testing_history[:,self.window_length:,4]
        # print("this is testing history")
        # print(testing_history)
        # cash_price = np.ones((1, -self.testing_end_time + self.testing_start_time))
        cash_price = np.ones((1, close_price.shape[1]))
        close_price = np.concatenate((cash_price, close_price), axis=0)
        close_price = np.transpose(close_price)
        self.close_price = close_price

        actualPV_list = np.sum(close_price * share_list, axis = 1)
        actualPV_list = np.insert(actualPV_list,0,1)
        self.actualPV_list = actualPV_list[:-1]


        price_list = np.asarray(price_list)
        relative_price = price_list / price_list[0]
        self.relative_price = relative_price
        self.average_return = np.mean(relative_price[:,1:],axis = 1)

        self.max_draw = max_drawdown(self.pv_list)
        self.alpha = alpha_value(self.pv_list, self.average_return)
        self.sharpe = sharpe_ratio(self.pv_list, self.average_return)

    def save_to_file(self):
        summary_data = {
             "model_id": self.config["model_id"],
             "weights": self.action_list.tolist(),
             "pv": self.pv_list.tolist(),
             "actual_pv": self.actualPV_list.tolist(),
             "share_change": self.share_change_list.tolist(),
             "share": self.share_list.tolist(),
             "close_price": self.close_price.tolist(),
             "relative_price": self.relative_price.tolist(),
             "average_return": self.average_return.tolist(),
             "date": self.time_list,
             "weight_dash": self.unadjust_weight_list.tolist(),
             "drawdown": self.max_draw,
             "alpha": self.alpha,
             "sharpe": self.sharpe,
             "stocks": self.testing_stocks}

        with open(self.summary_path+"summary.json",'w') as outfile:
            json.dump(summary_data, outfile, indent = 4 , sort_keys = True)

        return summary_data

    def save_to_db(self, summary):
        stocks = self.testing_stocks
        return_rate = summary["pv"][-1] - 1
        test_result = {"model_id":self.config["model_id"], "train_id": self.train_id, "stocks": stocks, "return_rate": return_rate,
                       "start_date": self.testing_start_time, "end_date": self.testing_end_time,
                       "summary": summary, "config":self.config}

        save_test_result_to_db(test_result)
        return test_result

    def start_session(self):
        tf.reset_default_graph()
        tf_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False,)
        self.sess = tf.Session(config = tf_config)
        tflearn.config.init_training_mode()

    def close_session(self):
        self.sess.close()


if __name__ == '__main__':

    parser = build_parser()
    args = vars(parser.parse_args())
    train_id = int(args['train_id'])

    # tf.reset_default_graph()
    # sess = tf.Session()
    # tflearn.config.init_training_mode()
    model = ddpg_restore_model(train_id)
    model.restore()
    model.backtest()

    if args["plot_weight"] == "True":
        plotWeight(model.action_list)
        
    if args["plot_pv"] == "True":
        name = ["DDPG","Validation","average_return"]
        figure_list = [model.pv_list,model.actualPV_list,model.average_return]
        plotPV(figure_list, name)
        name = ["stocks","average_return"]
        figure_list = [model.relative_price,model.average_return]
        plotPV(figure_list, name)
    print("task done !")


    




















































# batch_size = 64
# action_bound = 1.
# tau = 1e-3

# models = []
# model_names = []
# window_length_lst = [7]
# predictor_type_lst = ['cnn']
# use_batch_norm = True



# testing_stocks =  ['AAPL', 'ATVI', 'CMCSA']
# nb_classes = len(testing_stocks) + 1

# model_save_path_tmp = "train_package/1/weights/checkpoint.ckpt"
# summary_path_tmp = "train_package/1/results/"

# for window_length in window_length_lst:
#     for predictor_type in predictor_type_lst:
#         name = 'AAPL_stock_window_{}_predictor_{}'.format(window_length, predictor_type)
#         model_names.append(name)
#         tf.reset_default_graph()
#         sess = tf.Session()
#         tflearn.config.init_training_mode()
#         action_dim = [nb_classes]
#         state_dim = [nb_classes, window_length]
#         variable_scope = get_variable_scope(window_length, predictor_type, use_batch_norm)
#         with tf.variable_scope(variable_scope):
#             actor = StockActor(sess, state_dim, action_dim, action_bound, 1e-4, tau, batch_size, predictor_type, 
#                                use_batch_norm)
#             critic = StockCritic(sess=sess, state_dim=state_dim, action_dim=action_dim, tau=1e-3,
#                                  learning_rate=1e-3, num_actor_vars=actor.get_num_trainable_vars(), 
#                                  predictor_type=predictor_type, use_batch_norm=use_batch_norm)
#             actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

#             model_save_path = model_save_path_tmp
#             summary_path = summary_path_tmp

#             ddpg_model = DDPG(None, sess, actor, critic, actor_noise, obs_normalizer=obs_normalizer,
#                               config_file='config/stock.json', model_save_path=model_save_path,
#                               summary_path=summary_path)
#             ddpg_model.initialize(load_weights=True, verbose=False)
#             models.append(ddpg_model)





# history, abbreviation = read_stock_history(filepath='utils/datasets/stocks_history_target.h5')
# # history, abbreviation = read_stock_history(filepath='utils/datasets/AAPL_stock_history.h5')


# #history [stock, time, feature]
# history = history[:, :, :4]
# target_stocks = abbreviation[:3]
# print("target stocks are ", target_stocks)
# num_training_time = 1095
# num_testing_time = history.shape[1]
# print("training time and testing time are", num_training_time, num_testing_time)
# # window_length = int(args['window_length'])
# # including cash account

# # get target history
# target_history = np.empty(shape=(len(target_stocks), num_training_time, history.shape[2]))
# for i, stock in enumerate(target_stocks):
#     target_history[i] = history[abbreviation.index(stock), :num_training_time, :]



# # collect testing data

# testing_history = np.empty(shape=(len(testing_stocks), num_testing_time, history.shape[2]))
# for i, stock in enumerate(target_stocks):
#     testing_history[i] = history[abbreviation.index(stock), :num_testing_time, :]



# prev_pv = 1
# prev_action = np.zeros(shape=(1,nb_classes))
# prev_action[0][0] = 1

# action_list = []
# share_change_list = []
# share_list = []
# pv_list = []
# actualPV_list = []
# price_list =[]


# cost = 0.0025
# mu1_list = []


# for step in range(num_testing_time - window_length_lst[0]):
#     observation = testing_history[:, step:step + window_length_lst[0], :].copy()
#     cash_observation = np.ones((1, window_length_lst[0], observation.shape[2]))
#     observation = np.concatenate((cash_observation, observation), axis=0)
#     action = models[0].predict_single(observation)
#     # action_list.append(action)

#     date = step + window_length_lst[0]
#     today_price = testing_history[:, date-1, 3]
#     prev_price = testing_history[:, date-2, 3]
#     cash_price = np.ones((1,))

#     today_price = np.concatenate((cash_price, today_price), axis=0)
#     prev_price = np.concatenate((cash_price, prev_price), axis=0)

#     # print("shape of cash price", cash_price.shape)
#     # print("shape of today price", today_price.shape)


#     y_t = (today_price/prev_price)
#     # print("y_t shape", y_t.shape)
#     # print(y_t * prev_action)

#     unadjust_weight = (y_t.flatten() * prev_action.flatten()) / np.dot(y_t.flatten(), prev_action.flatten())
#     unadjust_pv = prev_pv * np.dot(y_t.flatten(), prev_action.flatten())
#     adjust_weight = action
#     mu1 = cost * (np.abs(unadjust_weight - adjust_weight)).sum()
#     adjust_pv = (1-mu1) * unadjust_pv
#     # adjust_pv = unadjust_pv

#     # potential mu need to be considered
#     share_change = (adjust_pv.flatten() * adjust_weight.flatten() - unadjust_pv.flatten() * unadjust_weight.flatten())\
#                    /today_price
#     # actualPV_change_list.append(np.dot(share_change, today_price))
#     # actualPV_change_list.append(np.sum(share_change * today_price))

    
#     # if step in [0,1,2,3]:
#     # 	print("step ",1)
#     # 	print("today's price is ", today_price)
#     # 	print("prev_price is ", prev_price)
#     # 	print("y_t is ", y_t)
#     # 	print("unadjust_weight", unadjust_weight)
#     # 	print("unadjust_pv", unadjust_pv)
#     # 	print("adjust_weight", adjust_weight)
#     # 	print("adjust_pv", adjust_pv)
#     # 	print("share_change ", share_change)
#     	# print("actualPV_change_list", actualPV_change_list[-1])
#     print(action)

#     pv_list.append(unadjust_pv)
#     share_change_list.append(share_change)
#     action_list.append(action)
#     price_list.append(today_price.flatten())
#     mu1_list.append(mu1)



#     prev_action = action
#     prev_pv = adjust_pv


# share_list = np.asarray(share_change_list)
# share_list[0][0] +=1
# share_list = np.cumsum(np.asarray(share_list), axis = 0)
# # print("share_list shape is",share_list.shape)

# close_price = testing_history[:,window_length_lst[0]:num_testing_time,3]
# cash_price = np.ones((1,num_testing_time - window_length_lst[0]))
# close_price = np.concatenate((cash_price, close_price), axis=0)
# close_price = np.transpose(close_price)
# actualPV_list = np.sum(close_price * share_list, axis = 1)
# actualPV_list = np.insert(actualPV_list,0,1) 






# price_list = np.asarray(price_list)
# price_list /= price_list[0]



# plotWeight(action_list)

# name = ["DDPG","stocks"]
# figure_list = [pv_list,price_list]

# plotPV(figure_list, name)


# # plt.figure()
# # plt.title("mu figure")
# # mu1_list = np.asarray(mu1_list).flatten()
# # x = list(range(mu1_list.shape[0]))
# # print("x",len(x),type(x))
# # print("mu1_list",mu1_list.shape, type(pv))
# # plt.plot(x, mu1_list,  linewidth=1.0, linestyle='-',label= "mu")
# # plt.show()



