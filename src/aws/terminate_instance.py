# Copyright 2018, Yintech Innovation Labs, All rights reserved.
# Written by Hanxiong Wang.

import boto3
import json
import os

def terminate_instance(aws_access_key_id = None, aws_secret_access_key = None, region_name = None):

	if aws_access_key_id and aws_secret_access_key and region_name:
		session = boto3.Session(aws_access_key_id = aws_access_key_id,
								aws_secret_access_key = aws_secret_access_key,
								region_name = region_name)
	else:
		session = boto3.Session()

	ec2 = session.resource('ec2')	

	with open("./tmp/instances_info.json", 'r') as infile:
		id_list = json.load(infile)["instances_id"]


	if (id_list == []):
		print("no ec2 instance to be terminated")
	else:
		for instance_id in id_list:
			instance = ec2.Instance(instance_id)
			response = instance.terminate()
			# print(response)
		print("close all created running instance")

	os.remove("./tmp/instances_info.json")
	
	return

if __name__ == '__main__':
	terminate_instance()
