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
import sys
import pickle


sys_host = "192.168.1.64"
pyRAPL.setup(devices=[pyRAPL.Device.PKG])




class Server:

	def __init__(self, server_id, server_type):
		self.workload = 0
		self.host = server_id

		if server_type == "cloud":
			self.batch = 8
		elif server_type == "edge":
			self.batch = 4
		else:
			print("Error: Unknown server type.")
			exit(1)

		self.type = server_type
		self.port = 5000

		self.image_reader = ImageReader()
		self.image_reader.load_state_dict(torch.load("image_reader.pth"))
		self.image_reader.eval()


	# convert the unit of time and energy consumption from micro
	def _convert_unit(self, t, e):
		t = t / 1e6
		e = e / 1e6
		return t, e

	# do inference
	def predict(self, task_dataset):
		measure = pyRAPL.Measurement('bar')
		measure.begin()


		result = []
		for image, _ in task_dataset:
			output = self.image_reader(image)
			prediction = torch.max(output, dim=1)
			result.append(prediction)

		measure.end()
		measure.end()

		t = measure.result.duration

		if measure.result.pkg != None:
			e = measure.result.pkg[0]
		else:
			e = 0.0

		t, e = self._convert_unit(t, e)
		return t, e

	def run(self):
		server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		server_socket.bind(("0.0.0.0", self.port))
		server_socket.listen(10)

		while True:
			print(f"{self.host}: starts to listen.", flush=True)

			client_socket, addr = server_socket.accept()

			print(f"{self.host}: client socket accepted.", flush=True)

			data_bytes = b""
			while True:
				chunk = client_socket.recv(4096)
				print(f"Got chunk of size {len(chunk)}", flush=True)
				if not chunk:
					break
				data_bytes += chunk

			print(f"{self.host}: data received.", flush=True)
			
			new_data = pickle.loads(data_bytes)

			sys_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			sys_socket.connect((sys_host, 5000))
			payload = json.dumps({"type": "server_status", "id": self.host, "workload": self.workload})
			sys_socket.send(payload.encode())
			sys_socket.close()

			print(f"{self.host}: server status 0 reported.", flush=True)

			loaded = DataLoader(dataset=new_data,  batch_size=1, shuffle=False)
			t, e = self.predict(loaded)

			print(f"{self.host}: inference processed.", flush=True)

			sys_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			sys_socket.connect((sys_host, 5000))
			payload = json.dumps({"type": "server_status", "id": self.host, "workload": self.workload})
			sys_socket.send(payload.encode())
			sys_socket.close()

			print(f"{self.host}: server status 1 reported.", flush=True)

			response = {"type": "task_result", "task_result": [t, e]}
			response_bytes = pickle.dumps(response)

			client_socket.sendall(response_bytes)
			client_socket.close()

			print(f"{self.host}: client replied", flush=True)



if __name__ == "__main__":
	device_id = (sys.argv[1])
	server_type = sys.argv[2]
	server = Server(device_id, server_type)
	server.run()