#!/bin/bash

# Stop and remove cloud container(s)
docker stop server_0
docker rm server_0

# Stop and remove edge containers
for i in 1 2; do
    docker stop server_$i
    docker rm server_$i
done

# Stop and remove client containers
for i in {0..9}; do
    docker stop client_$i
    docker rm client_$i
done

echo "All containers stopped and removed."
