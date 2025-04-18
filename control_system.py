import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from image_reader.image_reader import ImageReader
import time
import pyRAPL
import socket
import selectors
import random
import json
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import docker

class Sys(gym.Env):

	def __init__(self, num_clouds=1, num_edges=2, num_clients=10, cloud_thread=8,
			edge_thread=4, client_thread=1, max_steps=128, connection_status=None, verbose=False,
			action_discrete=True):
		super().__init__()

		self.num_clouds = num_clouds
		self.num_edges = num_edges
		self.num_clients = num_clients
		self.just_reset = False
		self.max_steps = max_steps

		self.verbose = verbose
		self.action_discrete = action_discrete


		self.port = 5000
		self.host = "0.0.0.0"

		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.socket.bind((self.host, self.port))
		self.socket.listen(20)



		self.threads = []
		for i in range(num_clouds):
			self.threads.append(cloud_thread)
		for i in range(num_edges):
			self.threads.append(edge_thread)
		for i in range(num_clients):
			self.threads.append(client_thread)
		self.workload = [0 for _ in range(num_clouds+num_edges+num_clients)]
		self.network_status = self.generate_random_network()


		temp_state = self.get_state(0,0,0)


		if self.action_discrete:
			self.action_space = spaces.Discrete(1+num_clouds+num_edges)
		else:
			self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1+num_clouds+num_edges,), dtype=np.float32)


		self.observation_space = spaces.Box(low=0, high=2**63-2, shape=temp_state.shape)


		self.client = docker.from_env()
		self.containers = []

		self.last_sock = None
		self.last_addr = None
		self.last_client = None
		self.request_not_replied = False

		self.reward = 0.0

		self.steps = 0
		self.itr = 0

		self.start_time = time.time()


	def generate_random_network(self):
		network = []
		for i in range(self.num_clients):
			connection = [1]
			connection.extend(random.choices(list(range(1, 11)), k=(self.num_clouds+self.num_edges-1)))
			random.shuffle(connection)
			network.append(connection)
		return network

	# update the reward with the task result
	# return the cumulative reward for current time frame
	def update_reward(self, task_size, energy, latency, latency_requirement, server_index):
		if latency > latency_requirement:
			latency_penalty = -task_size/10.0
		else:
			latency_penalty = 0.0

		energy_penalty = 1

		new_reward = task_size/10.0 - latency/10.0 - energy_penalty*energy/50.0 + latency_penalty

		self.reward = self.reward + new_reward

		return self.reward




	# return the cumulative reward for current time frame
	# set self.reward = 0
	def get_reward(self):
		reward = self.reward
		self.reward = 0.0
		return reward

	# return the observation
	def get_state(self, client_status, task_size, latency_requirement):
		client_id_encoded = [0] * self.num_clients
		client_id_encoded[client_status] = 1
		states = (
			self.threads +
			self.workload +
			sum(self.network_status, []) +
			client_id_encoded +
			[task_size] +
			[latency_requirement]
		)

		return np.array(states, dtype=np.float32)

	def pause(self):
		for name in self.containers:
			container = self.client.containers.get(name)
			container.pause()

	def unpause(self):
		for name in self.containers:
			container = self.client.containers.get(name)
			container.unpause()


	def step(self, action):
		self.steps = self.steps + 1

		if self.verbose:
			print(f"\tSteps: {self.steps}\n", flush=True)

		self.unpause()

		# reply decision if task_request is received in previous step
		if self.request_not_replied:

			if not self.action_discrete:
				action = np.argmax(action)

			if self.verbose:
				print(f"\t action: {action}", flush=True)

			if action == 0:
				expected_latency = 0
			else:
				expected_latency = self.network_status[self.last_client][action-1]

			decision = json.dumps({"type": "decision", "target_server": int(action), "expected_latency": expected_latency})
			self.last_sock.sendall(decision.encode('utf-8'))
			self.last_sock.close()
			self.request_not_replied = False

		while True:

			client_socket, client_addr = self.socket.accept()
			data = client_socket.recv(1024)
			request = json.loads(data.decode('utf-8'))

			if self.verbose:
				print(f"\t {request}", flush=True)

			# if task_request is received, end this step and return the observation and reward
			if request["type"] == "task_request":
				self.pause()
				self.request_not_replied = True
				client_id = int(request["id"].lstrip("client_"))
				task_size = float(request["size"])
				latency_requirement = float(request["latency_requirement"])
				self.last_client = client_id

				observation = self.get_state(client_id, task_size, latency_requirement)
				reward = self.get_reward()

				self.last_sock = client_socket
				self.last_addr = client_addr

				if self.steps >= self.max_steps:
					terminated = True
					self.steps = 0

					print(f"Itr {self.itr} done, has run for {time.time() - self.start_time}s.", flush=True)
					
					self.itr += 1
				else:
					terminated = False

				if self.verbose:
					print(f"\t\treward: {reward}")
				return observation, reward, terminated, False, {f"itr": self.itr}

			# update the reward
			elif request["type"] == "task_result":
				e = request["energy"]
				t = request["latency"]
				s = request["size"]
				t_req = request["latency_requirement"]
				server_index = request["server_index"]

				if self.verbose:
					print(f"task result: {request}", flush=True)

				self.update_reward(s, e, t, t_req, server_index)

				client_socket.close()

			# update the observation
			elif request["type"] == "server_status":

				if self.verbose:
					print(f"server status: {request}", flush=True)

				server = request["id"]

				if "server" in server:
					index = int(server.lstrip("server_"))
					self.workload[index] = float(request["workload"])
				elif "client"  in server:
					index = int(server.lstrip("client_")) + self.num_clouds + self.num_edges
					self.workload[index] = float(request["workload"])
				else:
					print("ID not found!")


				client_socket.close()
			else:
				client_socket.close()


	def reset(self, seed=None, options=None):

		if self.verbose:
			print("\nReset: system reset\n", flush=True)

		client = docker.from_env()
		network_name = "computation_offloading"


		for name in self.containers:
			try:
				container = self.client.containers.get(name)
				container.stop()
				container.remove()
			except Exception as e:
				print(f"Error cleaning up container {container.name}: {e}")

		self.containers = []


		for i in range(self.num_clouds):
			self.client.containers.run("server_image",
										name=f"server_{i}",
										network="computation_offloading",
										command=[f"server_{i}", "cloud"],
										detach=True,
										privileged=True,
										nano_cpus=4_000_000_000
										)
			container = self.client.containers.get(f"server_{i}")
			container.pause()
			self.containers.append(f"server_{i}")

		for i in range(self.num_edges):
			self.client.containers.run("server_image",
										name=f"server_{self.num_clouds + i}",
										network="computation_offloading",
										command=[f"server_{self.num_clouds + i}", "edge"],
										detach=True,
										privileged=True,
										nano_cpus=2_000_000_000
										)
			container = self.client.containers.get(f"server_{self.num_clouds + i}")
			container.pause()
			self.containers.append(f"server_{self.num_clouds + i}")

		for i in range(self.num_clients):
			self.client.containers.run("mobile_device_image",
										name=f"client_{i}",
										network="computation_offloading",
										command=f"client_{i}",
										detach=True,
										privileged=True,
										nano_cpus=1_000_000_000,
										)
			container = self.client.containers.get(f"client_{i}")
			container.pause()
			self.containers.append(f"client_{i}")


		# reinitialize the workload and network connection status
		self.workload = [0 for _ in range(self.num_clouds+self.num_edges+self.num_clients)]
		self.network_status = self.generate_random_network()

		observation = self.get_state(0,0,0)

		self.just_reset = True
		self.request_not_replied = False

		return observation, {"type": "reset"}

	# this function is required by gym, but we do not need this function.
	def render(self):
		...

	def close(self):

		for name in self.containers:
			try:
				container = self.client.containers.get(name)
				container.stop()
				container.remove()
			except Exception as e:
				print(f"Error cleaning up container {container.name}: {e}")


