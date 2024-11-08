import tensorflow as tf
import numpy as np
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.metrics import coco_evaluation
from object_detection.core import standard_fields
import os

# Load the TFLite model
model_path = 'models/retrained_ssdlite_mobiledet_td_20241106072657/ssdlite_mobiledet_cpu_320x320_td_qat_20000.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load label map
label_map_path = 'dataset/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

# Load test dataset
test_record_path = 'data/trash-detection/test.record'
dataset = tf.data.TFRecordDataset(test_record_path)

# Function to parse TFRecord
def parse_tfrecord(example_proto):
    features = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    image = tf.image.decode_jpeg(parsed_features['image/encoded'], channels=3)
    image = tf.image.resize(image, (320, 320))
    image = tf.cast(image, tf.uint8)
    return image, parsed_features

# Prepare dataset
dataset = dataset.map(parse_tfrecord)
dataset = dataset.batch(1)

# Initialize COCO evaluator
evaluator = coco_evaluation.CocoDetectionEvaluator(category_index)

# Evaluate model
for image, groundtruth in dataset:
    # Preprocess image
    input_data = np.array(image, dtype=np.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])

    # Prepare detections for evaluation
    detections = {
        standard_fields.DetectionResultFields.detection_boxes: boxes,
        standard_fields.DetectionResultFields.detection_classes: classes,
        standard_fields.DetectionResultFields.detection_scores: scores,
    }

    # Prepare groundtruth for evaluation
    groundtruth_boxes = tf.sparse.to_dense(groundtruth['image/object/bbox/xmin'])
    groundtruth_classes = tf.sparse.to_dense(groundtruth['image/object/class/label'])

    groundtruth_dict = {
        standard_fields.InputDataFields.groundtruth_boxes: groundtruth_boxes,
        standard_fields.InputDataFields.groundtruth_classes: groundtruth_classes,
    }

    # Add single image evaluation
    evaluator.add_single_ground_truth_image_info(image_id, groundtruth_dict)
    evaluator.add_single_detected_image_info(image_id, detections)

# Get evaluation metrics
metrics = evaluator.evaluate()
print("Evaluation metrics:", metrics)
