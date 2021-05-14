#!/bin/bash

echo "killing old docker processes"
docker-compose rm -fs

echo "building docker containers using BuildKit"
COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker-compose build
docker-compose up
