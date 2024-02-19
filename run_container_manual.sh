#!/bin/bash
container_name="pmaiedge"

if [ ! $(docker ps -q -f name=$container_name) ]; then
	docker rm $container_name
	docker run -u $(id -u):$(id -g) \
		   -it  \
		   -v ${PWD}:/home/PMAIEDGE \
		   --name $container_name \
		    pmaiedge
	echo "pmaiedge_debug was not running. Starting a new container."
else
	echo "The container is still running."
fi
