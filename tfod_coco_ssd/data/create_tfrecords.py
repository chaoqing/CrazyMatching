import os
import tensorflow as tf
from lxml import etree
import io
from PIL import Image # Pillow library for image processing

# --- Configuration ---
# Define your dataset paths
IMAGE_DIR = '/teamspace/studios/this_studio/Work/CrazyMatching/tfod_coco_ssd/data/images'
ANNOTATIONS_DIR = '/teamspace/studios/this_studio/Work/CrazyMatching/tfod_coco_ssd/data/annotations'
OUTPUT_TFRECORD_PATH = '/teamspace/studios/this_studio/Work/CrazyMatching/tfod_coco_ssd/data/train.record' # This will be updated for validation later

LABEL_MAP = {
    'animal_1010': 1,
    'animal_1016': 2,
    'animal_1089': 3,
    'animal_1236': 4,
    'animal_1400': 5,
    'animal_1478': 6,
    'animal_1492': 7,
    'animal_543': 8,
    'animal_639': 9,
    'animal_689': 10,
    'animal_806': 11,
    'animal_859': 12,
    'animal_913': 13,
    'animal_951': 14,
    'animal_986': 15
}

# --- Helper Functions ---

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def parse_voc_xml(xml_path):
    """Parses a Pascal VOC XML file and returns a dictionary of annotations."""
    tree = etree.parse(xml_path)
    root = tree.getroot()

    data = {
        'filename': root.find('filename').text,
        'width': int(root.find('size/width').text),
        'height': int(root.find('size/height').text),
        'objects': []
    }

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        xmin = float(obj.find('bndbox/xmin').text)
        ymin = float(obj.find('bndbox/ymin').text)
        xmax = float(obj.find('bndbox/xmax').text)
        ymax = float(obj.find('bndbox/ymax').text)

        data['objects'].append({
            'class_name': class_name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        })
    return data

def create_tf_example(image_path, annotation_data, label_map):
    """
    Creates a tf.train.Example from image and annotation data.
    Bounding box coordinates are normalized (0-1).
    """
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')

    width = annotation_data['width']
    height = annotation_data['height']
    filename = annotation_data['filename'].encode('utf8')

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes_text = []
    classes = []

    for obj in annotation_data['objects']:
        xmins.append(obj['xmin'] / width)
        ymins.append(obj['ymin'] / height)
        xmaxs.append(obj['xmax'] / width)
        ymaxs.append(obj['ymax'] / height)
        classes_text.append(obj['class_name'].encode('utf8'))
        classes.append(label_map[obj['class_name']])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(filename), # Often same as filename
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature(b'jpeg'),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))
    return tf_example

# --- Main Conversion Logic ---

def convert_pascal_to_tfrecord(image_dir, annotations_dir, output_path, label_map):
    writer = tf.io.TFRecordWriter(output_path)
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]

    for xml_file in xml_files:
        xml_path = os.path.join(annotations_dir, xml_file)
        annotation_data = parse_voc_xml(xml_path)

        image_filename = annotation_data['filename']
        image_path = os.path.join(image_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found for annotation {xml_file}. Skipping.")
            continue

        tf_example = create_tf_example(image_path, annotation_data, label_map)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print(f"Successfully created TFRecord file at: {output_path}")

if __name__ == '__main__':
    # Ensure you have tensorflow, lxml, and Pillow installed:
    # pip install tensorflow lxml Pillow

    # Generate training data first using generate_tfod_data.py
    # Then run this script to convert to TFRecord
    
    # For training data
    print("Converting training data to TFRecord...")
    convert_pascal_to_tfrecord(
        image_dir=IMAGE_DIR,
        annotations_dir=ANNOTATIONS_DIR,
        output_path=OUTPUT_TFRECORD_PATH,
        label_map=LABEL_MAP
    )
    print("Training TFRecord conversion complete.")
    
    # For validation data (assuming you generate validation data separately)
    # You would typically have a separate set of images and annotations for validation
    # For this example, we'll just use the same directories but output to a different file
    # In a real scenario, you'd split your dataset into train/val/test
    
    # print("\nConverting validation data to TFRecord...")
    # convert_pascal_to_tfrecord(
    #     image_dir=IMAGE_DIR, # Or your validation image directory
    #     annotations_dir=ANNOTATIONS_DIR, # Or your validation annotation directory
    #     output_path='/teamspace/studios/this_studio/Work/CrazyMatching/tfod_coco_ssd/data/val.record',
    #     label_map=LABEL_MAP
    # )
    # print("Validation TFRecord conversion complete.")
