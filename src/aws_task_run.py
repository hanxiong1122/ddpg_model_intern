#!/home/ubuntu/anaconda3/bin/python

# Copyright 2018, Yintech Innovation Labs, All rights reserved.
# Written by Hanxiong Wang.

import os
import boto3
import json
import time
from utils.s3utils import s3_service
from utils.modelutils import  load_config, get_result_path, load_aws_config



bucket_name = "hanxiong"

### connect to s3 ###########
# aws_config = load_aws_config()
# s3 = s3_service(aws_config["aws_access_key_id"], 
# 				aws_config["aws_secret_access_key"],
# 				aws_config["region_name"])
s3 = s3_service()
#### create a tmp folder if not exits ##############
if not os.path.exists("./tmp"):
	os.mkdir("./tmp")


while True:
	try:
		## for local test##
		# res = os.system("aws s3 mv s3://{bucket_name}/tasksconfig.json s3://{bucket_name}/tasksconfig_b.json".format(bucket_name = bucket_name))
		
		# for aws ec2 run
		res = os.system("/home/ubuntu/anaconda3/bin/aws s3 mv s3://{bucket_name}/tasksconfig.json s3://{bucket_name}/tasksconfig_b.json".format(bucket_name = bucket_name))

		flag = s3.download_and_del_file(bucket_name = bucket_name, 
								key = "tasksconfig_b.json", 
								filename = "./tmp/tasksconfig.json")
		if not flag:
			with open("./tmp/download_log.txt","a") as f:
				f.write("fail to download_and_del_file, code is " + str(res) + "\n")
			time.sleep(2)
			continue
		else:
			with open("./tmp/download_log.txt","a") as f:
				f.write("successfully download_and_del_file, code is " + str(res) + "\n")

		try:
			with open("./tmp/tasksconfig.json","r") as file:
				tasks_config = json.load(file)
			if len(tasks_config["stocks"]) != len(tasks_config["test_dates"]):
				with open("./tmp/error_log.txt","w") as f:
					f.write("stocks length and testdates length are not match")
					s3.put_file(bucket_name = bucket_name,
								path = "./tmp/error_log.txt",
								key = "error_log.txt")
				break

			if tasks_config["stocks"] == []:
				s3.put_file(bucket_name = bucket_name,
							path = "./tmp/tasksconfig.json",
							key = "tasksconfig.json")	
				break
			else:
				stock_string = tasks_config["stocks"].pop()
				test_start, test_end = tasks_config["test_dates"].pop()
				train_start, train_end = tasks_config["train_dates"]
				predictor_type = tasks_config["predictor_type"]
				window_length = str(tasks_config["window_length"])
				trading_cost = str(tasks_config["trading_cost"])
				episode = str(tasks_config["episode"])
				repeat_time = str(tasks_config["repeat_time"])
				feature_number = str(tasks_config["feature_number"])
				learning_rates = str(tasks_config["learning_rates"])
				activation_function = tasks_config["activation_function"]
				buffer_size = str(tasks_config["buffer size"])
				batch_size = str(tasks_config["batch size"])
				tasks_config["uncompleted_tasks_num"] = len(tasks_config["stocks"])

				with open("./tmp/working_log.txt","a") as f:
					f.write(stock_string + "\n")

				with open("./tmp/tasksconfig.json","w") as file:
					json.dump(tasks_config, file, indent = 4 , sort_keys = True)

				#### check if tasksconfig.json exists in s3 at this stage ###
				##  for local test ##
				# if 0 == os.system("aws s3 ls s3://{bucket_name}/tasksconfig.json".format(bucket_name = bucket_name)):

				## for aws ec2 run ##
				if 0 == os.system("/home/ubuntu/anaconda3/bin/aws s3 ls s3://{bucket_name}/tasksconfig.json".format(bucket_name = bucket_name)):
					with open("./tmp/error_log.txt","w") as f:
						f.write("at least two instances access and run the same task " + stock_string)
					s3.put_file(bucket_name = bucket_name,
								path = "./tmp/error_log.txt",
								key = "error_log.txt")
				
				s3.put_file(bucket_name = bucket_name,
							path = "./tmp/tasksconfig.json",
							key = "tasksconfig.json")

				## for local test##
				# cmd = "python aws_test_run.py" + \
				# 	 " -s " + stock_string + \
				# 	 " -ts " + test_start + \
				# 	 " -te " + test_end + \
				# 	 " -trs " + train_start + \
				# 	 " -tre " + train_end + \
				# 	 " -pt " + predictor_type + \
				# 	 " -wl " + window_length + \
				# 	 " -tc " + trading_cost + \
				# 	 " -ep " + episode + \
				# 	 " -rt " + repeat_time + \
				# 	 " -fn " + feature_number + \
				# 	 " -lr " + learning_rates + \
				# 	 " -af " + activation_function + \
				# 	 " -bs " + buffer_size + \
				# 	 " -bts " + batch_size

				# for aws ec2 run
				cmd = "./aws_test_run.py" + \
					 " -s " + stock_string + \
					 " -ts " + test_start + \
					 " -te " + test_end + \
					 " -trs " + train_start + \
					 " -tre " + train_end + \
					 " -pt " + predictor_type + \
					 " -wl " + window_length + \
					 " -tc " + trading_cost + \
					 " -ep " + episode + \
					 " -rt " + repeat_time + \
					 " -fn " + feature_number + \
					 " -lr " + learning_rates + \
					 " -af " + activation_function + \
					 " -bs " + buffer_size + \
					 " -bts " + batch_size
				os.system(cmd)

		except Exception as error:
			print(error)
			raise
	except Exception as error:
		print(error)

print("complete all the works")