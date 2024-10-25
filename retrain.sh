#!/bin/bash

# コンテナの実行
docker run --gpus all -it \
    -v $(pwd):/workspace/edge-trash-detection \  # マウントするディレクトリを指定
    edge-trash-detection \
    bash -c "cd /workspace/edge-trash-detection && python3 retrain_ssdlite_mobiledet.py"
