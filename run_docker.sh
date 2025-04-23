#!/bin/bash

# Check if GPU names are provided
if [ -z "$1" ]; then
  echo "Please provide GPU names (e.g., 0,1,2)"
  exit 1
fi

# GPU names passed as the first argument
GPU_NAMES=$1

# Run the Docker command with the specified GPUs. shm-size is needed for sharding, as Docker defaults to tiny RAM. Feel free to change this if it causes issues.
docker run -it --rm --gpus '"device='$GPU_NAMES'"' -v $(pwd):/rl_optimizer -w /rl_optimizer/rl_optimizer --shm-size=5g open