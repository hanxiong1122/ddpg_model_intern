# Copyright 2018, Yintech Innovation Lab, All rights reserved.
# Written by Hanxiong Wang.

import sys
import boto3
import json

def instance_terminare(instance_id_list):
	ec2 = boto3.resource('ec2')
	for instance_id in instance_id_list:
	    instance = ec2.Instance(instance_id)
	    response = instance.terminate()
	    print(response)

def list_instance():
	ec2 = boto3.resource('ec2')
	for instance in ec2.instances.all():
	    print(instance.id, instance.state)
	return [ins.id for ins in ec2.instances.all()], [ins.state for ins in ec2.instances.all()]


def load_cmdconfig():
	try:
		with open("./config/cmdconfig.json") as f:
			return json.load(f)["cmd"]
	except:
		print("no cmd config file")
		raise

def load_aws_config(config_path="./config/aws_config.json"):
    try:
        with open(config_path) as f:
            config = json.load(f)
            print("successfully load config file at " + config_path)
        return config
    except:
        print("error aws config file or path")
        raise


def load_instances_ip():
	try:
		with open("./tmp/instances_info.json") as f:
			return json.load(f)["public_ip_addresses"]
	except:
		print("no instances info file")
		raise