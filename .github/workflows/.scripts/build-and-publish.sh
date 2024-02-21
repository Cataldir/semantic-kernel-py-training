#!/bin/bash

# Define variables
IMAGE_NAME="skapp"
TAG="v1.0"
REGISTRY="amlsandbox.azurecr.io"

# Build the Docker image
docker build -t $IMAGE_NAME:$TAG .

# Tag the Docker image
docker tag $IMAGE_NAME:$TAG $REGISTRY/$IMAGE_NAME:$TAG

# Push the Docker image to the Container Registry
docker push $REGISTRY/$IMAGE_NAME:$TAG