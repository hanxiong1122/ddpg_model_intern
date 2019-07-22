import pandas as pd 
import os
import json
import logging
import time
from utils.modelutils import  load_config
from stock_model_train import ddpg_trading_train
from stock_backtest import ddpg_restore_model

DEBUG = False

# config = load_config()
# config["run_id"] = 0  # this is will be overridden when driven from UI
# train_id = ddpg_trading_train(config, DEBUG = DEBUG).train_model()
# model = ddpg_restore_model(train_id)
# model.restore()
# model.backtest()
# print("Finish work on %s"%(train_id))
# print("Finish all the work")




# target_list = ['AAPL', 'ATVI', 'CMCSA', 'COST', 'CSX', 'DISH', 'EA', 'EBAY', 'FB', 'GOOGL', 'HAS', 'ILMN', 'INTC',
#                'MAR', 'REGN', 'SBUX']



stocks = [["MSFT"]]#, ["AMD"]]#,"AMD","ABIO","ACC"]
predictor_types = ["lstm"]
window_lengths = [3]
trading_costs = [0]
episodes = [25]
activation_functions =["prelu"] #["relu","tanh","leaky_relu","prelu"]
repeat_time = 1
feature_numbers = [5]
reward_functions = [2]
learning_rates = [0.001]

for stock in stocks:
	for predictor_type in predictor_types:
		for window_length in window_lengths:
			for trading_cost in trading_costs:
				for activation_function in activation_functions:
					for episode in episodes:
						for feature_number in feature_numbers:
							for _ in range(repeat_time):
								for learning_rate in learning_rates:
									config = load_config()
									config["model_id"] = 0  #this is will be overridden when driven from UI
									config["input"]["feature_number"] = feature_number
									config["input"]["window_length"] = window_length
									config["input"]["predictor_type"] = predictor_type
									config["input"]["trading_cost"] = trading_cost
									config["training"]["episode"] = episode
									config["input"]["stocks"] = stock
									config["layers"]["activation_function"] = activation_function
									config["training"]["actor learning rate"] = 0.001
									config["training"]["critic learning rate"] = 0.001
									config["training"]["buffer size"] = 100000
									config["training"]["max_step"] = 0
									config["training"]["batch size"] = 64
									print(config)
									train_id = ddpg_trading_train(config, DEBUG = DEBUG).train_model()
									model = ddpg_restore_model(train_id)
									model.restore()
									model.backtest()
									print("Finish work on %s"%(train_id))






