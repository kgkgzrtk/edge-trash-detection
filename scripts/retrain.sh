#!/bin/bash

docker run --gpus all -it \
    -v $(pwd):/workspace/edge-trash-detection \
    edge-trash-detection \
    bash -c "cd /workspace/edge-trash-detection && python3 scripts/retrain_ssdlite_mobiledet.py"
