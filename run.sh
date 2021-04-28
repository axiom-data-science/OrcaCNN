#!/bin/bash

echo "killing old docker processes"
sudo docker-compose rm -fs

echo "building docker containers using BuildKit"
sudo COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker-compose build