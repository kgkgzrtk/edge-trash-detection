import os
import json
import tensorflow as tf
from pycocotools.coco import COCO
from object_detection.utils import dataset_util

def create_tf_example(image_info, annotations, image_dir, category_name):
    image_path = os.path.join(image_dir, image_info['file_name'])
    
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image = fid.read()
    
    width = image_info['width']
    height = image_info['height']
    
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    
    for ann in annotations:
        xmin = ann['bbox'][0] / width
        ymin = ann['bbox'][1] / height
        xmax = (ann['bbox'][0] + ann['bbox'][2]) / width
        ymax = (ann['bbox'][1] + ann['bbox'][3]) / height
        
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
        
        category_id = ann['category_id']
        classes.append(category_id)
        classes_text.append(category_name[category_id].encode('utf8'))
    
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

def create_label_map(category_name, output_file):
    with open(output_file, 'w') as f:
        for category_id, category in category_name.items():
            f.write("item {\n")
            f.write(f"  id: {int(category_id)}\n")
            f.write(f"  name: '{category}'\n")
            f.write("}\n\n")

def main():
    # TACOデータセットのパスを修正
    taco_dir = 'data/TACO'
    output_dir = 'dataset'
    os.makedirs(output_dir, exist_ok=True)
    annotations_file = os.path.join(taco_dir, 'data/annotations.json')
    
    coco = COCO(annotations_file)
    
    # カテゴリ名の取得
    category_name = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}
    
    # ラベルマップの作成
    label_map_output = os.path.join(output_dir, 'label_map.pbtxt')
    create_label_map(category_name, label_map_output)
    
    # 訓練データとテストデータの分割（80:20）
    image_ids = list(coco.imgs.keys())
    train_size = int(len(image_ids) * 0.8)
    
    train_ids = image_ids[:train_size]
    test_ids = image_ids[train_size:]
    
    # 訓練データの作成
    train_writer = tf.io.TFRecordWriter(os.path.join(output_dir, 'train.record'))
    for image_id in train_ids:
        image_info = coco.imgs[image_id]
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
        tf_example = create_tf_example(image_info, annotations, os.path.join(taco_dir, 'data'), category_name)
        train_writer.write(tf_example.SerializeToString())
    train_writer.close()
    
    # テストデータの作成
    test_writer = tf.io.TFRecordWriter(os.path.join(output_dir, 'test.record'))
    for image_id in test_ids:
        image_info = coco.imgs[image_id]
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
        tf_example = create_tf_example(image_info, annotations, os.path.join(taco_dir, 'data'), category_name)
        test_writer.write(tf_example.SerializeToString())
    test_writer.close()

if __name__ == '__main__':
    main()