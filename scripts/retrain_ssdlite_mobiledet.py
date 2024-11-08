import os
import sys
import argparse
from datetime import datetime
import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2

# Set environment variables
os.environ['PYTHONPATH'] += ':/tensorflow/models/research/:/tensorflow/models/research/slim/'

# Check for experiment name argument; if not provided, use a default with timestamp
parser = argparse.ArgumentParser(description='Retrain SSDLite MobileDet model')
parser.add_argument('--experiment_name', type=str, 
                    default=f"retrained_ssdlite_mobiledet_td_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    help='Name of the experiment')
parser.add_argument('--resume', type=str, default=None,
                    help='Path to the checkpoint to resume training from')
args = parser.parse_args()

# Set output directory based on whether resuming or not
if args.resume:
    OUTPUT_DIR = os.path.join('models', f"{args.experiment_name}_resumed")
    print(f"Resuming training from checkpoint: {args.resume}")
else:
    OUTPUT_DIR = os.path.join('models', args.experiment_name)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# Dataset and output directory setup
DATA_DIR = 'dataset'

# Model setup
PRETRAINED_MODEL_NAME = 'ssdlite_mobiledet_cpu_320x320_coco_2020_05_19'
PRETRAINED_MODEL_DIR = os.path.join('models', 'pretrained_model')
os.makedirs(PRETRAINED_MODEL_DIR, exist_ok=True)
PRETRAINED_MODEL_PATH = os.path.join(PRETRAINED_MODEL_DIR, PRETRAINED_MODEL_NAME)
PRETRAINED_MODEL_TAR = os.path.join(PRETRAINED_MODEL_DIR, f'{PRETRAINED_MODEL_NAME}.tar.gz')

# Download and extract the pretrained model if necessary
if not os.path.exists(PRETRAINED_MODEL_PATH):
    if not os.path.exists(PRETRAINED_MODEL_TAR):
        os.system(f'wget http://download.tensorflow.org/models/object_detection/{PRETRAINED_MODEL_NAME}.tar.gz -O {PRETRAINED_MODEL_TAR}')
    os.system(f'tar -xzvf {PRETRAINED_MODEL_TAR} -C {PRETRAINED_MODEL_DIR}')

# Training parameters
NUM_STEPS = 40000  # Increase training steps
BATCH_SIZE = 16    # Adjust batch size
NUM_CLASSES = 5   # Trash Detection dataset class count

# Use the pre-configured pipeline config
CONFIG_PATH = 'config/ssdlite_mobilenet_td.config'  # Use the provided config file

# Load and update pipeline config with specific training parameters
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:
    text_format.Merge(f.read(), pipeline_config)

# Update pipeline config with training parameters
pipeline_config.model.ssd.num_classes = NUM_CLASSES
pipeline_config.train_config.batch_size = BATCH_SIZE
pipeline_config.train_config.num_steps = NUM_STEPS
pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.total_steps = NUM_STEPS
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(
    PRETRAINED_MODEL_PATH,
    'model.ckpt-400000'
)
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_config.from_detection_checkpoint = True
# Enable ssdlite, this should already be enabled in the config we downloaded, but this is just to make sure.
pipeline_config.model.ssd.box_predictor.convolutional_box_predictor.kernel_size = 3
pipeline_config.model.ssd.box_predictor.convolutional_box_predictor.use_depthwise = True
pipeline_config.model.ssd.feature_extractor.use_depthwise = True
# Quantization Aware Training
pipeline_config.graph_rewriter.quantization.delay = 0
pipeline_config.graph_rewriter.quantization.weight_bits = 8
pipeline_config.graph_rewriter.quantization.activation_bits = 8

# Save modified pipeline.config
config_text = text_format.MessageToString(pipeline_config)
OUTPUT_CONFIG_PATH = os.path.join(OUTPUT_DIR, 'pipeline.config')
with tf.io.gfile.GFile(OUTPUT_CONFIG_PATH, "w") as f:
    f.write(config_text)
print(f"Modified pipeline.config saved at {OUTPUT_CONFIG_PATH}")

# TensorBoard log directory setup
TENSORBOARD_LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

# Update fine_tune_checkpoint if resuming
if args.resume:
    pipeline_config.train_config.fine_tune_checkpoint = args.resume

# Run training
start = datetime.now()
train_command = f'python3 /tensorflow/models/research/object_detection/model_main.py \
    --model_dir={OUTPUT_DIR} \
    --pipeline_config_path={OUTPUT_CONFIG_PATH} \
    --num_train_steps={NUM_STEPS} \
    --alsologtostderr \
    --log_dir={TENSORBOARD_LOG_DIR}'
print(f"Running training command: {train_command}")
train_result = os.system(train_command)
if train_result != 0:
    raise RuntimeError("Training failed.")

# Training duration
end = datetime.now()
duration = end - start
hours, remainder = divmod(duration.seconds, 3600)
minutes, seconds = divmod(remainder, 60)
print(f'TRAINING TIME: {hours}:{minutes:02d}:{seconds:02d}')