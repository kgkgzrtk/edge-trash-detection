import os
import json
import tensorflow as tf
from pycocotools.coco import COCO
from object_detection.utils import dataset_util

def create_tf_example(image_info, annotations, image_dir, label_name, top_labels):
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
        original_label_id = ann['category_id']
        if original_label_id not in top_labels:
            continue
        
        xmin = ann['bbox'][0] / width
        ymin = ann['bbox'][1] / height
        xmax = (ann['bbox'][0] + ann['bbox'][2]) / width
        ymax = (ann['bbox'][1] + ann['bbox'][3]) / height
        
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
        
        original_label_name = label_name[original_label_id]
        
        classes.append(original_label_id)
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

def main():
    taco_dir = 'data/TACO'
    output_dir = 'dataset'
    os.makedirs(output_dir, exist_ok=True)
    annotations_file = os.path.join(taco_dir, 'data/annotations.json')
    
    coco = COCO(annotations_file)
    
    label_name = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}
    
    image_ids = list(coco.imgs.keys())
    train_size = int(len(image_ids) * 0.8)
    
    train_ids = image_ids[:train_size]
    test_ids = image_ids[train_size:]
    
    label_count = {cat_id: 0 for cat_id in label_name.keys()}
    
    for image_id in image_ids:
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
        for ann in annotations:
            label_count[ann['category_id']] += 1
    
    # 上位10個のクラスを取得
    top_labels = sorted(label_count, key=label_count.get, reverse=True)[:10]
    
    # 新しいIDを1から順にマッピング
    id_mapping = {cat_id: idx + 1 for idx, cat_id in enumerate(top_labels)}
    
    # label_map.pbtxtの作成
    with open(os.path.join(output_dir, 'label_map.pbtxt'), 'w') as f:
        for cat_id, new_id in id_mapping.items():
            f.write(f"item {{\n  id: {new_id}\n  name: '{label_name[cat_id]}'\n}}\n\n")
    
    train_writer = tf.io.TFRecordWriter(os.path.join(output_dir, 'train.record'))
    for image_id in train_ids:
        image_info = coco.imgs[image_id]
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
        tf_example = create_tf_example(image_info, annotations, os.path.join(taco_dir, 'data'), label_name, id_mapping)
        train_writer.write(tf_example.SerializeToString())
    train_writer.close()
    
    val_writer = tf.io.TFRecordWriter(os.path.join(output_dir, 'val.record'))
    for image_id in test_ids:
        image_info = coco.imgs[image_id]
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
        tf_example = create_tf_example(image_info, annotations, os.path.join(taco_dir, 'data'), label_name, id_mapping)
        val_writer.write(tf_example.SerializeToString())
    val_writer.close()
    
    print(f"Total number of labels: {len(top_labels)}")
    print(f"Number of training images: {len(train_ids)}")
    print(f"Number of validation images: {len(test_ids)}")
    print(f"Train/Validation split ratio: {len(train_ids) / len(image_ids):.2f}/{len(test_ids) / len(image_ids):.2f}")
    
    for cat_id in top_labels:
        print(f"Label '{label_name[cat_id]}': {label_count[cat_id]} images")

if __name__ == '__main__':
    main()