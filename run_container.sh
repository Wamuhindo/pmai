#!/bin/bash
container_name="pmaiedge"
 

if [ ! $(docker ps -q -f name=$container_name) ]; then
	use_gpu = 0
	if ["$1" = "--gpu"]; then
		use_gpu = 1
	docker rm $container_name
	docker run -u $(id -u):$(id -g) \
		   -it  \
		   -v ${PWD}:/home/PMAIEDGE \
		   --name $container_name --gpus all \
		   -d pmaiedge python3 main_checkpoint.py -c all
	echo "pmaiedge was not running. Starting a new container."
else
	echo "The container is still running."
fi
