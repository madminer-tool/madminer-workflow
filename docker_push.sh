#!/bin/bash
docker login -u "$DOCKERUSER" -p "$DOCKERPASS"
docker build -f docker/docker-madminer-physics/Dockerfile -t madminertool/docker-madminer-physics .
docker push madminertool/docker-madminer-physics
docker build -f docker/docker-madminer-ml/Dockerfile -t madminertool/docker-madminer-ml .
docker push madminertool/docker-madminer-ml