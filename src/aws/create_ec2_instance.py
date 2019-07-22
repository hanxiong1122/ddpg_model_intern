# Copyright 2018, Yintech Innovation Labs, All rights reserved.
# Written by Hanxiong Wang.

import boto3
import time
import json
import os

# def create_instance()

class create_ec2_instances(object):
	def __init__(self, 
				aws_access_key_id = None, 
				aws_secret_access_key = None, 
				region_name = None,
				ImageId = 'ami-c7eaf7b8', 
				MaxCount = 1, 
				MinCount = 1, 
				InstanceType = 't2.micro', 
				time_out = 30,
				user_data = None,
				Arn = None):
		try: 
			if aws_access_key_id and aws_secret_access_key and region_name:
				self.session = boto3.Session(aws_access_key_id = aws_access_key_id,
											  aws_secret_access_key = aws_secret_access_key,
											  region_name = region_name)
			else:
				self.session = boto3.Session()
			self.ec2 = self.session.resource('ec2')
		except Exception as error:
			print("can not connect to aws ec2")
			raise
		self.ImageId = ImageId
		self.MaxCount = MaxCount
		self.MinCount = MinCount
		self.InstanceType = InstanceType
		self.time_out = time_out
		self.user_data = user_data
		self.Arn = Arn
		self.public_ip_address_list, self.id_list = self.create_running_instance()

	def create_running_instance(self):
		instances = self.ec2.create_instances(ImageId = self.ImageId,
												MinCount = self.MinCount,
												MaxCount = self.MaxCount,
												InstanceType= self.InstanceType,
												UserData = self.user_data,
												IamInstanceProfile = {
													'Arn': self.Arn
													# 'Name': 'AiWorker'
												})

		running_instance_num = 0
		id_list = [ins.id for ins in instances]
		created_instance_num = len(id_list)
		print("created ", created_instance_num," instances ids, they are ", id_list)

		# a collection of instances with id_list
		instance_collection = self.ec2.instances.filter(InstanceIds = id_list)
		# filter criterion, need find running instance
		filter = [{'Name': 'instance-state-name', 'Values': ['running']}]

		start_time = time.time()
		while (running_instance_num != created_instance_num): 
			filtered_collection = instance_collection.filter(Filters = filter)
			filtered_instances = [_ for _ in filtered_collection]
			running_instance_num = len(filtered_instances)
			print("running_instance_num is ", running_instance_num)
			time.sleep(5)
			end_time = time.time()
			if ((end_time - start_time) > self.time_out * 5): 
				break

		print("We created ", running_instance_num, "running instances")

		return self.save_info(filtered_instances)	

	def save_info(self, filtered_instances):
		for instance in filtered_instances:
			print("instance id is ", instance.id, "public_ip_address is ", instance.public_ip_address)
		public_ip_addresses = [instance.public_ip_address for instance in filtered_instances]
		instances_ids = [instance.id for instance in filtered_instances]

		if not os.path.isdir("./tmp/"):
			os.mkdir("./tmp")
		if not os.path.exists("./tmp/instances_info.json"):
			instances_info = { 
							"public_ip_addresses": public_ip_addresses,
							"instances_id": instances_ids
							}
			with open("./tmp/instances_info.json",'w') as outfile:
				json.dump(instances_info, outfile, indent = 4 , sort_keys = True)
		else:
			with open("./tmp/instances_info.json",'r') as infile:
				instances_info = json.load(infile)
				for public_ip_address in public_ip_addresses:
					instances_info["public_ip_addresses"].append(public_ip_address)
				for instance_id in instances_ids:
					instances_info["instances_id"].append(instance_id)
			with open("./tmp/instances_info.json",'w') as outfile:
				json.dump(instances_info, outfile, indent = 4, sort_keys = True)
		return public_ip_addresses, instances_ids


	def terminate(self):
		if (self.id_list == []):
			print("no ec2 instance to be terminated")
			return

		for instance_id in self.id_list:
			instance = self.ec2.Instance(instance_id)
			response = instance.terminate()
			print(response)
		self.id_list = []
		print("close all created running instance")
		return

	def get_instance_public_ip(self):
		return self.public_ip_address_list

	def get_instance_id(self):
		return self.id_list

if __name__=="__main__":
	ImageId = 'ami-095f89c443d0824b5'
	instances = create_ec2_instances(ImageId = ImageId, MaxCount = 2, MinCount = 1, InstanceType = 'm4.xlarge')
	print(instances.get_instance_public_ip())
	print(instances.get_instance_id())

