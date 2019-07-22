import pandas as pd 
import numpy as np
import os
import json
import logging
import time
import shutil
from utils.modelutils import  load_config, get_result_path
from stock_model_train import ddpg_trading_train
from stock_backtest import ddpg_restore_model
import bs4 as bs 
import pickle 
import requests
import random 



DEBUG = False
portfolio_size = 30
sample_times = 3

if __name__ == '__main__':

	def save_sp500_tickers():
		resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
		soup = bs.BeautifulSoup(resp.text, 'lxml')
		table = soup.find('table', {'class': 'wikitable sortable'})
		tickers = []
		for row in table.findAll('tr')[1:]:
			ticker = row.findAll('td')[0].text
			tickers.append(ticker)

		with open("sp500tickers.pickle","wb") as f:
			pickle.dump(tickers,f)

		return tickers


	stocks = []
	testdates = []
	dates = ["2017-07-24","2018-07-27"]
	#get all SP500 stocks ticker from wikipedia 
	SP500_tickers = save_sp500_tickers()

	#set random numbers 
	for i in range(sample_times):
		index = random.sample(range(len(SP500_tickers)-1),30)
		index = np.sort(index)
		portfolio_ticker = [SP500_tickers[i] for i in index ]
		stocks.append(portfolio_ticker)
		testdates.append(dates)


	predictor_type = "lstm"
	window_length = 3
	trading_cost = 0
	episode = 1
	activation_function = "prelu" #["relu","tanh","leaky_relu","prelu"]
	repeat_time = 1
	feature_number = 5
	reward_function = 2
	learning_rates = 0.001


	config = load_config()
	config["model_id"] = 0  # this is will be overridden when driven from UI
	config["input"]["feature_number"] = feature_number
	config["input"]["window_length"] = window_length
	config["input"]["predictor_type"] = predictor_type
	config["input"]["trading_cost"] = trading_cost
	config["training"]["episode"] = episode
	config["layers"]["activation_function"] = activation_function
	config["training"]["actor learning rate"] = learning_rates
	config["training"]["critic learning rate"] = learning_rates
	config["training"]["buffer size"] = 20000
	config["training"]["max_step"] = 0
	config["training"]["batch size"] = 64



	for i, stock in enumerate(stocks):
		config["input"]["stocks"] = stock
		config["testing"]["testing_start_time"] = testdates[i][0]
		config["testing"]["testing_end_time"] = testdates[i][1]
		config["training"]["max_step"] = 0
		best_pv = -1
		best_id = -1
		for i in range(repeat_time):
			train_id = ddpg_trading_train(config, DEBUG = DEBUG).train_model()
			model = ddpg_restore_model(train_id)
			model.restore()
			model.backtest()
			result_path = get_result_path(train_id) + "summary.json"
			print("Finish work on %s"%(train_id))
			with open(result_path,'r') as outfile:
				f = json.load(outfile)
			if best_pv <= f["pv"][-1]:
				best_pv = f["pv"][-1]
				best_id = train_id
		print("scanning best model at ",best_id)
		if os.path.exists('train_package/{}'.format("_".join(stock))):
			shutil.rmtree("{}".format('train_package/{}'.format("_".join(stock))))
		shutil.copytree('train_package/{}'.format(best_id),
						'train_package/{}'.format("_".join(stock)))
		for i in range(repeat_time):
			shutil.rmtree("train_package/{}".format(i))
	print("Finish work on %s"%("_".join(stock)))


