#!/bin/bash

# Get the latest model directory
LATEST_MODEL_DIR=$(ls -td models/retrained_ssdlite_mobiledet_td_* | head -n 1)

if [ -z "$LATEST_MODEL_DIR" ]; then
    echo "No trained model found in models directory"
    exit 1
fi

echo "Using model directory: $LATEST_MODEL_DIR"

# Run quantization in Docker container
docker run --gpus all -it --rm \
    -v $(pwd):/workspace/edge-trash-detection \
    edge-trash-detection \
    bash -c "
        cd /workspace/edge-trash-detection && \
        python3 scripts/quantize_model.py --model_dir=$LATEST_MODEL_DIR
    " 