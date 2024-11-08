#!/bin/bash

# Define the number of steps as a variable
NUM_STEPS=31013

# Get the latest model directory
#LATEST_MODEL_DIR=$(ls -td models/retrained_ssdlite_mobiledet_td_* | head -n 1)
LATEST_MODEL_DIR="models/retrained_ssdlite_mobiledet_td_20241107064141"
if [ -z "$LATEST_MODEL_DIR" ]; then
    echo "No trained model found in models directory"
    exit 1
fi

echo "Using model directory: $LATEST_MODEL_DIR"

# Run export and quantize in Docker container (TF1)
echo "Starting export and quantization process..."
docker run --gpus all -it --rm \
    -v $(pwd):/workspace/edge-trash-detection \
    edge-trash-detection \
    bash -c "
        cd /workspace/edge-trash-detection && \
        export PYTHONPATH=/tensorflow/models/research:/tensorflow/models/research/slim:\$PYTHONPATH && \
        python3 /tensorflow/models/research/object_detection/export_tflite_ssd_graph.py \
            --pipeline_config_path=/workspace/edge-trash-detection/${LATEST_MODEL_DIR}/pipeline.config \
            --trained_checkpoint_prefix=/workspace/edge-trash-detection/${LATEST_MODEL_DIR}/model.ckpt-${NUM_STEPS} \
            --output_directory=/workspace/edge-trash-detection/${LATEST_MODEL_DIR}/frozen_graph \
            --add_postprocessing_op=true
    "
if [ $? -ne 0 ]; then
    echo "Error during export and quantization process."
    exit 1
fi
echo "Export and quantization process completed successfully."

# Use a specific Docker image for tflite_convert
echo "Starting TFLite conversion..."
docker run --rm -v $(pwd):/workspace tensorflow/tensorflow:2.4.1 \
    bash -c "
        tflite_convert \
            --output_file=/workspace/${LATEST_MODEL_DIR}/ssdlite_mobiledet_cpu_320x320_td_qat_${NUM_STEPS}.tflite \
            --graph_def_file=/workspace/${LATEST_MODEL_DIR}/frozen_graph/tflite_graph.pb \
            --inference_type=QUANTIZED_UINT8 \
            --input_arrays='normalized_input_image_tensor' \
            --output_arrays='TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3' \
            --mean_values=128 \
            --std_dev_values=128 \
            --input_shapes=1,320,320,3 \
            --allow_custom_ops \
            --enable_v1_converter
    "
if [ $? -ne 0 ]; then
    echo "Error during TFLite conversion."
    exit 1
fi
echo "TFLite conversion completed successfully."