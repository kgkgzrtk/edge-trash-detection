#!/bin/bash

# コンテナの実行
docker run --gpus all -it \
    -v $(pwd)/../:/workspace/edge-trash-detection \
    edge-trash-detection \
    bash -c "cd /workspace/edge-trash-detection/eval && python3 evaluate.py"
