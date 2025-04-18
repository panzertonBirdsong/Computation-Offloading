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
import pickle
import sys

device = torch.device("cpu")

# pyRAPL.setup()
pyRAPL.setup(devices=[pyRAPL.Device.PKG])

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

sys_host = "192.168.1.64"
cloud_host = "server_0"
server_host = "server_"

class MobileDevice:


	def __init__(self, device_id=0):
		self.id = device_id
		self.workload = 0

		self.image_reader = ImageReader()
		self.image_reader.load_state_dict(torch.load("image_reader.pth"))
		self.image_reader.eval()

		self.host = "client_" + str(device_id)
		self.port = 5000

		self.task_index = 0

	# convert the unit of time and energy consumption from micro
	def _convert_unit(self, t, e):
		t = t / 1e6
		e = e / 1e6
		return t, e



	def predict(self, task_dataset):
		measure = pyRAPL.Measurement('bar')
		measure.begin()


		result = []
		for image, _ in task_dataset:
			output = self.image_reader(image)
			prediction = torch.max(output, dim=1)
			result.append(prediction)

		measure.end()

		t = measure.result.duration

		if measure.result.pkg != None:
			e = measure.result.pkg[0]
		else:
			e = 0.0

		t, e = self._convert_unit(t, e)
		
		return t, e

	def _generate_task(self):
		size = random.randint(0,1000)
		new_data = []
		for i in range(size):
			index = random.randint(0, len(data)-1)
			new_data.append(data[index])

		t = size * 0.1
		t_req = 0.5 * t if random.random() < 0.5 else t

		return new_data, size, t_req

	'''
	Send the status to the control system
	'''
	def _get_decision(self, size, latency):
		client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		client_socket.connect((sys_host, 5000))
		payload = json.dumps({"type": "task_request", "id": self.host, "size": size, "latency_requirement": latency, "task_index": self.task_index})
		client_socket.send(payload.encode())

		response = client_socket.recv(1024).decode()
		client_socket.close()


		response = json.loads(response)
		assert response["type"] == "decision"
		decision = response["target_server"]
		expected_latency = response["expected_latency"]
		return int(decision), expected_latency


	'''
	Send the result (reward)
	'''
	def _send_result(self, t, e, size, latency, server_id):
		client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		client_socket.connect((sys_host, 5000))
		payload = json.dumps({"type": "task_result", "id": self.host, "energy": e, "latency": t, "size": size, "latency_requirement": latency, "server_index": server_id, "task_index": self.task_index})
		client_socket.send(payload.encode())
		client_socket.close()
		

	def run(self):
		

		while True:
			self.task_index = self.task_index + 1

			time.sleep(random.random())




			new_data, new_size, new_latency = self._generate_task()

			print(f"{self.host}: task generated {self.task_index}.", flush=True)

			
			decision, expected_latency = self._get_decision(new_size, new_latency)
			start_time = time.time()

			if decision == 0:
				server_index = -1
				print(f"{self.host}: decision received on 0 {self.task_index}.", flush=True)

				self.workload = new_size

				client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
				client_socket.connect((sys_host, 5000))
				payload = json.dumps({"type": "server_status", "id": self.host, "workload": self.workload})
				client_socket.send(payload.encode())
				client_socket.close()




				loaded = DataLoader(dataset=new_data,  batch_size=1, shuffle=False)
				t, e = self.predict(loaded)

				print(f"{self.host}: local processing finished {self.task_index}.", flush=True)

				self.workload = self.workload - new_size

				client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
				client_socket.connect((sys_host, 5000))
				payload = json.dumps({"type": "server_status", "id": self.host, "workload": self.workload})
				client_socket.send(payload.encode())
				client_socket.close()

				print(f"{self.host}: client server status reported {self.task_index}.", flush=True)

			else:
				print(f"{self.host}: decision received on {decision}-1 {self.task_index}.", flush=True)

				expected_latency = expected_latency * 5 + random.randint(-5, 10)
				time.sleep(max(0.1, expected_latency))

				server_index = decision
				decision = decision - 1
				serialized = pickle.dumps(new_data)
				with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

					s.connect((f"server_{str(decision)}", 5000))
					s.sendall(serialized)
					s.shutdown(socket.SHUT_WR)

					print(f"{self.host}: data send to server {self.task_index}.", flush=True)

					# Receive reply
					reply_bytes = b""
					while True:
						chunk = s.recv(4096)
						if not chunk:
							break
						reply_bytes += chunk

					reply = pickle.loads(reply_bytes)

					print(f"{self.host}: data received from server {self.task_index}.", flush=True)

					assert reply["type"] == "task_result"
					[t, e] = reply["task_result"]


			end_time = time.time()
			processing_time = end_time - start_time

			print(f"{self.host}: report task result to sys {self.task_index}.", flush=True)

			self._send_result(processing_time, e, new_size, new_latency, server_index)



def test():
	mobile_device = MobileDevice()


	transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
	test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
	test_loader  = DataLoader(dataset=test_dataset,  batch_size=64, shuffle=False)
	mobile_device.predict(test_loader)



if __name__ == '__main__':

	device_id = int(sys.argv[1].lstrip("client_"))
	mobile_device = MobileDevice(device_id=device_id)
	mobile_device.run()