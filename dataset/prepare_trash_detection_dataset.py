import os
import json
import tensorflow as tf
from object_detection.utils import dataset_util
import subprocess
from pycocotools.coco import COCO

def download_and_prepare_dataset(url, output_dir):
    # Download the dataset
    dataset_dir = os.path.join(output_dir, 'trash-detection')
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Path to the ZIP file
    zip_path = os.path.join(dataset_dir, 'dataset.zip')
    
    # Download the ZIP file if it doesn't exist
    if not os.path.exists(zip_path):
        download_command = f"wget -O {zip_path} {url}"
        subprocess.run(download_command, shell=True, check=True)
    
    # Unzip the file
    unzip_command = f"unzip -n {zip_path} -d {dataset_dir}"
    subprocess.run(unzip_command, shell=True, check=True)

    # Count images in each folder
    for folder in ['train', 'test', 'valid']:
        image_dir = os.path.join(dataset_dir, folder)
        num_images = len([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        print(f"Number of images in {folder}: {num_images}")

    # Path to the annotations file for each folder
    annotations_files = {
        'train': os.path.join(dataset_dir, 'train', '_annotations.coco.json'),
        'test': os.path.join(dataset_dir, 'test', '_annotations.coco.json'),
        'valid': os.path.join(dataset_dir, 'valid', '_annotations.coco.json')
    }

    # Load the first annotation file to get label information
    coco = COCO(annotations_files['train'])

    # Create label map
    label_name = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}
    unique_labels = {cat['name']: cat['id'] for cat in coco.loadCats(coco.getCatIds())}

    # Create a mapping for IDs starting from 1
    id_mapping = {cat['id']: idx + 1 for idx, cat in enumerate(coco.loadCats(coco.getCatIds()))}

    with open(os.path.join(output_dir, 'label_map.pbtxt'), 'w') as f:
        for cat in coco.loadCats(coco.getCatIds()):
            new_id = id_mapping[cat['id']]
            f.write(f"item {{\n  id: {new_id}\n  name: '{cat['name']}'\n}}\n\n")

    # Function to create TFRecord example
    def create_tf_example(image_info, annotations, image_dir):
        # Ground Truth Boxesの制限を解除するため、以下の修正を追加
        max_detections = 300  # デフォルトの100から300に変更
        
        # 画像の読み込みと基本的な検証
        image_path = os.path.join(image_dir, image_info['file_name'])
        if not os.path.exists(image_path):
            return None

        with tf.io.gfile.GFile(image_path, 'rb') as fid:
            encoded_image = fid.read()

        # 整数値に変換
        width = int(image_info['width'])    # floatからintに変換
        height = int(image_info['height'])  # floatからintに変換

        # バウンディングボックスの正規化を確実に行う
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for ann in annotations:
            # バウンディングボックスの座標を正規化（0-1の範囲に収める）
            xmin = max(0.0, min(1.0, ann['bbox'][0] / float(width)))
            ymin = max(0.0, min(1.0, ann['bbox'][1] / float(height)))
            xmax = max(0.0, min(1.0, (ann['bbox'][0] + ann['bbox'][2]) / float(width)))
            ymax = max(0.0, min(1.0, (ann['bbox'][1] + ann['bbox'][3]) / float(height)))

            # 正規化された座標の検証
            if xmin >= xmax or ymin >= ymax:
                print(f"Warning: Invalid box coordinates for image {image_info['file_name']}")
                continue

            xmins.append(xmin)
            xmaxs.append(xmax)
            ymins.append(ymin)
            ymaxs.append(ymax)

            original_label_id = ann['category_id']
            original_label_name = label_name[original_label_id]

            # Use the mapped ID for the class
            mapped_label_id = id_mapping[original_label_id]
            classes.append(mapped_label_id)
            classes_text.append(original_label_name.encode('utf8'))

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(image_info['file_name'].encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(str(image_info['id']).encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_image),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example

    # Initialize counts for each folder and class
    folder_counts = {folder: {name: 0 for name in unique_labels.keys()} for folder in ['train', 'test', 'valid']}
    total_images_count = 0

    # Write TFRecord files for each folder
    for folder in ['train', 'test', 'valid']:
        tfrecord_path = os.path.join(output_dir, f'{folder}.record')
        writer = tf.io.TFRecordWriter(tfrecord_path)

        # Load annotations for the current folder
        coco_folder = COCO(annotations_files[folder])
        
        # Correct the path by using the correct directory structure
        image_dir = os.path.join(dataset_dir, folder)
        print(f"Checking images in directory: {image_dir}")

        # Get image IDs for the current folder
        image_ids = coco_folder.getImgIds()
        total_images_count += len(image_ids)

        for image_id in image_ids:
            image_info = coco_folder.imgs[image_id]
            annotations = coco_folder.loadAnns(coco_folder.getAnnIds(imgIds=image_id))
            tf_example = create_tf_example(image_info, annotations, image_dir)
            if tf_example is not None:
                writer.write(tf_example.SerializeToString())
                for ann in annotations:
                    original_label_id = ann['category_id']
                    class_name = label_name.get(original_label_id, None)
                    if class_name is None:
                        print(f"Warning: Class ID {original_label_id} not found in label_name")
                        continue
                    folder_counts[folder][class_name] += 1

        writer.close()
        print(f"TFRecord for {folder} has been saved to {tfrecord_path}")

    # Output dataset summary
    print(f"Total number of images: {total_images_count}")
    print(f"Total number of classes: {len(unique_labels)}")
    for name, idx in unique_labels.items():
        print(f"Class '{name}': {len(coco.getImgIds(catIds=[idx]))} images")

    # Print table of counts
    print("\nDataset Split Counts:")
    print(f"{'Split':<10} " + " ".join([f"{name:<10}" for name in unique_labels.keys()]))
    for folder, counts in folder_counts.items():
        print(f"{folder:<10} " + " ".join([f"{counts[name]:<10}" for name in unique_labels.keys()]))

# Example usage
download_and_prepare_dataset(
    url="https://universe.roboflow.com/ds/4NrFduZWyx?key=EHIxafiPlc",
    output_dir="data/trash-detection"
)
