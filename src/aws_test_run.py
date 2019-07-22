#!/home/ubuntu/anaconda3/bin/python

# Copyright 2018, Yintech Innovation Labs, All rights reserved.
# Written by Hanxiong Wang.

import pandas as pd 
import os
import json
import logging
import time
import shutil
import argparse

from utils.modelutils import  load_config, get_result_path, load_aws_config
from stock_model_train import ddpg_trading_train
from stock_backtest import ddpg_restore_model
from utils.s3utils import s3_service

DEBUG = False
bucket_name = "hanxiong"


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='provide stocks to be predicted')
	parser.add_argument('--stocks', '-s', help='give stocks', required = True)
	parser.add_argument('--test_start', '-ts', help='give test start date', required = True)
	parser.add_argument('--test_end', '-te', help='give test end date', required = True)
	parser.add_argument('--train_start', '-trs', help='give training start date,', required = True)
	parser.add_argument('--train_end', '-tre', help='give train end date', required = True)
	parser.add_argument('--predictor_type', '-pt', help='give predictor_type', required = True)
	parser.add_argument('--window_length', '-wl', help='give window_length',required = True)
	parser.add_argument('--trading_cost', '-tc', help='give trading_cost', required = True)
	parser.add_argument('--episode', '-ep', help='give episode', required = True)
	parser.add_argument('--repeat_time', '-rt', help='give repeat_time', required = True)
	parser.add_argument('--feature_number', '-fn', help='give feature_number', required = True)
	parser.add_argument('--learning_rates', '-lr', help='give learning_rates', required = True)
	parser.add_argument('--activation_function', '-af', help='give activation_function', required = True)
	parser.add_argument('--buffer_size', '-bs', help='give buffer size', required = True)
	parser.add_argument('--batch_size', '-bts', help='give batch size', required = True)


	args = vars(parser.parse_args())

	if args['stocks']:
		print(args['stocks'])
		stocks = [args['stocks'].split(",")]
		print(stocks)

	if args['test_start'] and args['test_end']:
		test_dates = [[args['test_start'], args['test_end']]]

	if args['train_start'] and args['train_end']:
		train_dates = [args['train_start'], args['train_end']]

	if args['predictor_type']:
		predictor_type = args['predictor_type']

	if args['window_length']:
		window_length = int(args['window_length'])

	if args['trading_cost']:
		trading_cost = int(args['trading_cost'])

	if args['episode']:
		episode = int(args['episode'])

	if args['repeat_time']:
		repeat_time = int(args['repeat_time'])

	if args['feature_number']:
		feature_number = int(args['feature_number'])

	if args['learning_rates']:
		learning_rates = float(args['learning_rates'])

	if args['activation_function']:
		activation_function = args['activation_function']

	if args['buffer_size']:
		buffer_size = int(args['buffer_size'])

	if args['batch_size']:
		batch_size = int(args['batch_size'])

	config = load_config()
	config["model_id"] = 0  # this is will be overridden when driven from UI
	config["input"]["training_start_time"] = train_dates[0]
	config["input"]["training_end_time"] = train_dates[1]
	config["input"]["feature_number"] = feature_number
	config["input"]["window_length"] = window_length
	config["input"]["predictor_type"] = predictor_type
	config["input"]["trading_cost"] = trading_cost
	config["training"]["episode"] = episode
	config["layers"]["activation_function"] = activation_function
	config["training"]["actor learning rate"] = learning_rates
	config["training"]["critic learning rate"] = learning_rates
	config["training"]["buffer size"] = buffer_size
	config["training"]["max_step"] = 0
	config["training"]["batch size"] = batch_size

	## connect to s3
	# aws_config = load_aws_config()
	# s3 = s3_service(aws_config["aws_access_key_id"], 
	# 				aws_config["aws_secret_access_key"],
	# 				aws_config["region_name"])
	s3 = s3_service()

	## start computing
	for i, stock in enumerate(stocks):
		config["input"]["stocks"] = stock
		config["testing"]["testing_start_time"] = test_dates[i][0]
		config["testing"]["testing_end_time"] = test_dates[i][1]
		config["input"]["eval_start_time"] = test_dates[i][0]
		config["input"]["eval_end_time"] = test_dates[i][1]
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
			
		## create a zip file and upload to s3
		shutil.make_archive(base_name = './train_package/{}'.format("_".join(stock)),
							format = 'zip',
							root_dir = './train_package/{}'.format("_".join(stock)))
		s3.put_file(bucket_name = bucket_name,
					path = './train_package/{}.zip'.format("_".join(stock)),
					key = "{}.zip".format("_".join(stock)))
		print("Finish work on %s"%("_".join(stock)))
