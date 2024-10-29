#!/bin/bash

docker run -it \
    -v $(pwd):/workspace/edge-trash-detection \
    -p 6006:6006 \
    edge-trash-detection \
    bash -c "
        cd /workspace/edge-trash-detection && \
        tensorboard --logdir=models --host=0.0.0.0 --port=6006
    "
