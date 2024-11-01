#!/bin/bash

# Experiment name argument with default timestamp
EXPERIMENT_NAME=${1:-"retrained_ssdlite_mobiledet_td_$(date +%Y%m%d%H%M%S)"}

docker run --gpus all -it --shm-size=1g --ulimit memlock=-1\
    -v $(pwd):/workspace/edge-trash-detection \
    edge-trash-detection \
    bash -c "
        cd /workspace/edge-trash-detection && \
        python3 scripts/retrain_ssdlite_mobiledet.py --experiment_name $EXPERIMENT_NAME
    "
