#!/bin/bash

docker rmi -f mobile_device_image
docker rmi -f server_image

cd mobile_device || { echo "mobile_device directory not found"; exit 1; }
docker build -t mobile_device_image .

cd ../server || { echo "server directory not found"; exit 1; }
docker build -t server_image .
