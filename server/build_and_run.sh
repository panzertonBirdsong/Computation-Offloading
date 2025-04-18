docker build -t server_image .

docker network create edge_network

docker run -d --rm \
  --network edge_network \
  --name server_1 \
  server_image device_1