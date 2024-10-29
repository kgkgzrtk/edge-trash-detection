import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# 環境変数の設定
os.environ['PYTHONPATH'] += ':/tensorflow/models/research/:/tensorflow/models/research/slim/'

# データセットの準備
DATA_DIR = '/data/TACO/data'
OUTPUT_DIR = '/output_ssdlite_mobiledet_taco'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# モデルの準備
PRETRAINED_MODEL_DIR = '/pretrained_model'
os.makedirs(PRETRAINED_MODEL_DIR, exist_ok=True)
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19.tar.gz'
PRETRAINED_MODEL_TAR = os.path.join(PRETRAINED_MODEL_DIR, 'ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19.tar.gz')

# モデルのダウンロード
if not os.path.exists(PRETRAINED_MODEL_TAR):
    os.system(f'wget {PRETRAINED_MODEL_URL} -O {PRETRAINED_MODEL_TAR}')
    os.system(f'tar -xvf {PRETRAINED_MODEL_TAR} -C {PRETRAINED_MODEL_DIR}')

# 訓練パラメータの設定
NUM_STEPS = 10000
BATCH_SIZE = 32
NUM_CLASSES = 60  # TACOデータセットの���テゴリ数に合わせて変更

# コンフィグファイルの設定
CONFIG_PATH = '/tensorflow/models/research/object_detection/samples/configs/ssdlite_mobiledet_edgetpu_320x320_coco_sync_4x4.config'

# ファイルの存在確認
if not tf.io.gfile.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(DATA_DIR, 'train.record')]
pipeline_config.train_input_reader.label_map_path = os.path.join(DATA_DIR, 'label_map.pbtxt')
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(DATA_DIR, 'val.record')]
pipeline_config.eval_input_reader[0].label_map_path = os.path.join(DATA_DIR, 'label_map.pbtxt')
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(PRETRAINED_MODEL_DIR, 'ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19/fp32/model.ckpt')
pipeline_config.train_config.batch_size = BATCH_SIZE
pipeline_config.train_config.num_steps = NUM_STEPS
pipeline_config.model.ssd.num_classes = NUM_CLASSES

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:
    f.write(config_text)

# モデルの訓練
os.system(f'python3 /tensorflow/models/research/object_detection/model_main.py \
    --logtostderr=true \
    --model_dir=/content/train \
    --pipeline_config_path={CONFIG_PATH}')

# モデルのエクスポート
os.system(f'python3 /tensorflow/models/research/object_detection/export_tflite_ssd_graph.py \
    --pipeline_config_path={CONFIG_PATH} \
    --trained_checkpoint_prefix=/content/train/model.ckpt-{NUM_STEPS} \
    --output_directory={OUTPUT_DIR} \
    --add_postprocessing_op=true')

# TFLiteモデルの量子化と変換
converter = tf.lite.TFLiteConverter.from_saved_model(OUTPUT_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# TFLiteモデルの保存
tflite_model_path = os.path.join(OUTPUT_DIR, 'model_quantized.tflite')
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f'Quantized TFLite model saved at {tflite_model_path}')

# モデルの量子化プロセス
def quantize_model(saved_model_dir, output_tflite_model):
    # Converter to TFLite model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert the model
    tflite_model = converter.convert()

    # Save the model
    with open(output_tflite_model, 'wb') as f:
        f.write(tflite_model)

# 量子化用のパス設定
SAVED_MODEL_DIR = os.path.join(OUTPUT_DIR, 'saved_model')
TFLITE_MODEL_PATH = os.path.join(OUTPUT_DIR, 'model_quantized.tflite')

# モデル量子化の実行
quantize_model(SAVED_MODEL_DIR, TFLITE_MODEL_PATH)
print(f"量子化されたモデルが {TFLITE_MODEL_PATH} に保存されました。")
