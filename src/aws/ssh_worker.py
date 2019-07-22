# Copyright 2018, Yintech Innovation Labs, All rights reserved.
# Written by Hanxiong Wang.

import boto3
import botocore
import paramiko
from utils.utils import load_cmdconfig, load_instances_ip

class ssh_worker(object):
	def __init__(self, key_path, instance_ip, cmd_list = None, username = "ubuntu"):
		self.key = paramiko.RSAKey.from_private_key_file(key_path)
		self.instance_ip = instance_ip
		self.cmd_list = cmd_list
		self.username = username
		self.client = paramiko.SSHClient()
		self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	
	def establish_connect(self):
		try:
			self.client.connect(hostname = self.instance_ip, username = self.username, pkey = self.key)
		except Exception as e:
			print(e)

	def run_cmd(self):
		for cmd in self.cmd_list:
			stdin, stdout, stderr = self.client.exec_command(cmd)
	
	def run_single_cmd(self, cmd):
		stdin, stdout, stderr = self.client.exec_command(cmd)
		stdout.channel.recv_exit_status()
		return stdout.readlines()
	
	def close_connection(self):
		self.client.close()




if __name__=="__main__":
	instances_ip = load_instances_ip()
	cmd_list = load_cmdconfig()
	for i, instance_ip in enumerate(instances_ip):
		worker = ssh_worker(key_path = "yintech.pem", 
							instance_ip = instance_ip, 
							cmd_list = cmd_list)
		worker.establish_connect()
		print("run the ", i, "th cmd")
		worker.run_cmd()
		worker.close_connection()

