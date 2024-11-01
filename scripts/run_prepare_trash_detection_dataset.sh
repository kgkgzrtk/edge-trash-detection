#!/bin/bash

# Dockerコンテナを起動してPythonスクリプトを実行
docker run --rm -it \
    -v $(pwd):/workspace/edge-trash-detection \
    edge-trash-detection \
    bash -c "
        cd /workspace/edge-trash-detection && \
        python3 dataset/prepare_trash_detection_dataset.py
    " 