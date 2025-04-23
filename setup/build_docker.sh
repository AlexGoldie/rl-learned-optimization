#!/bin/bash

# Set default image name
IMAGE_NAME="open"

# Get current user info
USER_NAME=$(id -un)
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Check if WANDB_API key is provided
if [ -z "$1" ]; then
  echo "Please provide WANDB_API_KEY key as argument"
  exit 1
fi
WANDB_API=$1

# Build the Docker image with current user info
docker build \
  --build-arg USERNAME="${USER_NAME}" \
  --build-arg USER_UID="${USER_ID}" \
  --build-arg USER_GID="${GROUP_ID}" \
  --build-arg WANDB_API="${WANDB_API}" \
  -t ${IMAGE_NAME} \
  .