from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf
from PIL import Image
from StringIO import StringIO
import glob
import random

import input_preprocess

def num_examples_per_epoch(split='train'):
    if split == 'train':
        return 118000
    elif split == 'val':
        return 5000
    else:
        assert(0)

def num_classes(only_objects=True):
    if only_objects:
        return 81
    else:
        return 133

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

id2trainid_objects = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11,
                      14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20,
                      23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
                      35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38,
                      44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47,
                      54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56,
                      63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65,
                      76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74,
                      86: 75, 87: 76, 88: 77, 89: 78, 90: 79}

id2trainid = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11,
              14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20,
              23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
              35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38,
              44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47,
              54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56,
              63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65,
              76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74,
              86: 75, 87: 76, 88: 77, 89: 78, 90: 79, 92: 80, 93: 81, 95: 82, 100: 83,
              107: 84, 109: 85, 112: 86, 118: 87, 119: 88, 122: 89, 125: 90, 128: 91,
              130: 92, 133: 93, 138: 94, 141: 95, 144: 96, 145: 97, 147: 98, 148: 99,
              149: 100, 151: 101, 154: 102, 155: 103, 156: 104, 159: 105, 161: 106,
              166: 107, 168: 108, 171: 109, 175: 110, 176: 111, 177: 112, 178: 113,
              180: 114, 181: 115, 184: 116, 185: 117, 186: 118, 187: 119, 188: 120,
              189: 121, 190: 122, 191: 123, 192: 124, 193: 125, 194: 126, 195: 127,
              196: 128, 197: 129, 198: 130, 199: 131, 200: 132}

id2name = {0: u'person', 1: u'bicycle', 2: u'car', 3: u'motorcycle', 4: u'airplane',
           5: u'bus', 6: u'train', 7: u'truck', 8: u'boat', 9: u'traffic light',
           10: u'fire hydrant', 11: u'stop sign', 12: u'parking meter', 13: u'bench',
           14: u'bird', 15: u'cat', 16: u'dog', 17: u'horse', 18: u'sheep', 19: u'cow',
           20: u'elephant', 21: u'bear', 22: u'zebra', 23: u'giraffe', 24: u'backpack',
           25: u'umbrella', 26: u'handbag', 27: u'tie', 28: u'suitcase', 29: u'frisbee',
           30: u'skis', 31: u'snowboard', 32: u'sports ball', 33: u'kite', 34: u'baseball bat',
           35: u'baseball glove', 36: u'skateboard', 37: u'surfboard', 38: u'tennis racket',
           39: u'bottle', 40: u'wine glass', 41: u'cup', 42: u'fork', 43: u'knife',
           44: u'spoon', 45: u'bowl', 46: u'banana', 47: u'apple', 48: u'sandwich',
           49: u'orange', 50: u'broccoli', 51: u'carrot', 52: u'hot dog', 53: u'pizza',
           54: u'donut', 55: u'cake', 56: u'chair', 57: u'couch', 58: u'potted plant',
           59: u'bed', 60: u'dining table', 61: u'toilet', 62: u'tv', 63: u'laptop',
           64: u'mouse', 65: u'remote', 66: u'keyboard', 67: u'cell phone',
           68: u'microwave', 69: u'oven', 70: u'toaster', 71: u'sink', 72: u'refrigerator',
           73: u'book', 74: u'clock', 75: u'vase', 76: u'scissors', 77: u'teddy bear',
           78: u'hair drier', 79: u'toothbrush', 80: u'banner', 81: u'blanket',
           82: u'bridge', 83: u'cardboard', 84: u'counter', 85: u'curtain',
           86: u'door-stuff', 87: u'floor-wood', 88: u'flower', 89: u'fruit',
           90: u'gravel', 91: u'house', 92: u'light', 93: u'mirror-stuff', 94: u'net',
           95: u'pillow', 96: u'platform', 97: u'playingfield', 98: u'railroad',
           99: u'river', 100: u'road', 101: u'roof', 102: u'sand', 103: u'sea',
           104: u'shelf', 105: u'snow', 106: u'stairs', 107: u'tent', 108: u'towel',
           109: u'wall-brick', 110: u'wall-stone', 111: u'wall-tile', 112: u'wall-wood',
           113: u'water-other', 114: u'window-blind', 115: u'window-other',
           116: u'tree-merged', 117: u'fence-merged', 118: u'ceiling-merged',
           119: u'sky-other-merged', 120: u'cabinet-merged', 121: u'table-merged',
           122: u'floor-other-merged', 123: u'pavement-merged', 124: u'mountain-merged',
           125: u'grass-merged', 126: u'dirt-merged', 127: u'paper-merged', 128: u'food-other-merged',
           129: u'building-other-merged', 130: u'rock-merged', 131: u'wall-other-merged', 132: u'rug-merged'}

id2color = [[0, 0, 0],
            [220, 20, 60], [119, 11, 32], [0, 0, 142], [0, 0, 230], [106, 0, 228],
            [0, 60, 100], [0, 80, 100], [0, 0, 70], [0, 0, 192], [250, 170, 30],
            [100, 170, 30], [220, 220, 0], [175, 116, 175], [250, 0, 30],
            [165, 42, 42], [255, 77, 255], [0, 226, 252], [182, 182, 255],
            [0, 82, 0], [120, 166, 157], [110, 76, 0], [174, 57, 255],
            [199, 100, 0], [72, 0, 118], [255, 179, 240], [0, 125, 92],
            [209, 0, 151], [188, 208, 182], [0, 220, 176], [255, 99, 164],
            [92, 0, 73], [133, 129, 255], [78, 180, 255], [0, 228, 0],
            [174, 255, 243], [45, 89, 255], [134, 134, 103], [145, 148, 174],
            [255, 208, 186], [197, 226, 255], [171, 134, 1], [109, 63, 54],
            [207, 138, 255], [151, 0, 95], [9, 80, 61], [84, 105, 51],
            [74, 65, 105], [166, 196, 102], [208, 195, 210], [255, 109, 65],
            [0, 143, 149], [179, 0, 194], [209, 99, 106], [5, 121, 0],
            [227, 255, 205], [147, 186, 208], [153, 69, 1], [3, 95, 161],
            [163, 255, 0], [119, 0, 170], [0, 182, 199], [0, 165, 120],
            [183, 130, 88], [95, 32, 0], [130, 114, 135], [110, 129, 133],
            [166, 74, 118], [219, 142, 185], [79, 210, 114], [178, 90, 62],
            [65, 70, 15], [127, 167, 115], [59, 105, 106], [142, 108, 45],
            [196, 172, 0], [95, 54, 80], [128, 76, 255], [201, 57, 1],
            [246, 0, 122], [191, 162, 208], [255, 255, 128], [147, 211, 203],
            [150, 100, 100], [168, 171, 172], [146, 112, 198], [210, 170, 100],
            [92, 136, 89], [218, 88, 184], [241, 129, 0], [217, 17, 255],
            [124, 74, 181], [70, 70, 70], [255, 228, 255], [154, 208, 0],
            [193, 0, 92], [76, 91, 113], [255, 180, 195], [106, 154, 176],
            [230, 150, 140], [60, 143, 255], [128, 64, 128], [92, 82, 55],
            [254, 212, 124], [73, 77, 174], [255, 160, 98], [255, 255, 255],
            [104, 84, 109], [169, 164, 131], [225, 199, 255], [137, 54, 74],
            [135, 158, 223], [7, 246, 231], [107, 255, 200], [58, 41, 149],
            [183, 121, 142], [255, 73, 97], [107, 142, 35], [190, 153, 153],
            [146, 139, 141], [70, 130, 180], [134, 199, 156], [209, 226, 140],
            [96, 36, 108], [96, 96, 96], [64, 170, 64], [152, 251, 152],
            [208, 229, 228], [206, 186, 171], [152, 161, 64], [116, 112, 0],
            [0, 114, 143], [102, 102, 156], [250, 141, 255]]

def convert_to_tfrecords(img_path, seg_path, out_path, split, encoded=True):
    img_list = glob.glob(os.path.join(img_path, '*.jpg'))
    random.seed(0)
    random.shuffle(img_list)

    total = len(img_list)

    tfrecord_file_name = os.path.join(out_path, 'coco_{}.tfrecords'.format(split))
    writer = tf.python_io.TFRecordWriter(tfrecord_file_name)

    count = 0
    for im_file_name in img_list:
        print(im_file_name)
        im_data = tf.gfile.FastGFile(im_file_name, 'r').read()
        image_id = im_file_name.split('/')[-1].split('.')[0]

        label_file_name = os.path.join(seg_path, image_id + '.png')
        print(label_file_name)
        print(count, total)

        label_data = tf.gfile.FastGFile(label_file_name, 'r').read()

        im = np.asarray(Image.open(StringIO(im_data)))
        if len(im.shape) < 3:
            continue
        rows, cols, depth = im.shape[0], im.shape[1], im.shape[2]
        if not encoded:
            im_data = np.asarray(Image.open(StringIO(im_data))).tostring()

            label_data = np.asarray(Image.open(StringIO(label_data)))
            label_data = label_data.tostring()
        else:
            label_data = np.asarray(Image.open(StringIO(label_data)))
            label_out = StringIO()
            Image.fromarray(label_data).save(label_out, 'png')
            label_data = label_out.getvalue()

        example = tf.train.Example(features=tf.train.Features(feature={
                                    'height': _int64_feature(rows),
                                    'width' : _int64_feature(cols),
                                    'depth' : _int64_feature(depth),
                                    'image' : _bytes_feature(im_data),
                                    'labels': _bytes_feature(label_data),
                                    'id' : _bytes_feature(image_id),
                                   }))

        writer.write(example.SerializeToString())
        count = count + 1

    writer.close()

def get_dataset(filename,
                buffer_size = 100,
                batch_size = 4,
                num_epochs = 50,
                num_parallel_calls = 8,
                encoded = True,
                crop_height = 512,
                crop_width = 512,
                resize_shape = 512,
                data_augment = False):

    dataset = tf.data.TFRecordDataset([filename])
    def parser(record):
        keys_to_features = {'height': tf.FixedLenFeature((), tf.int64),
                            'width' : tf.FixedLenFeature((), tf.int64),
                            'depth' : tf.FixedLenFeature((), tf.int64),
                            'image' : tf.FixedLenFeature((), tf.string),
                            'labels' : tf.FixedLenFeature((), tf.string),
                            'id' : tf.FixedLenFeature((), tf.string),
                           }
        parsed = tf.parse_single_example(record, keys_to_features)

        if encoded:
            image = tf.image.decode_image(parsed['image'])
            labels = tf.image.decode_image(parsed['labels'])
        else:
            image = tf.decode_raw(parsed['image'], tf.uint8)
            labels = tf.decode_raw(parsed['labels'], tf.uint8)

        height = tf.cast(parsed['height'], tf.int32)
        width = tf.cast(parsed['width'], tf.int32)
        #depth = tf.cast(parsed['depth'], tf.int32)

        image_shape = tf.stack([height, width, 3])
        labels_shape = tf.stack([height, width, 1])

        image = tf.reshape(image, image_shape)
        labels = tf.reshape(labels, labels_shape)

        _, image, labels = \
                input_preprocess.preprocess_image_and_label(image, labels,
                                                            crop_height, crop_width,
                                                            min_resize_value = resize_shape,
                                                            max_resize_value = resize_shape,
                                                            min_scale_factor=1.0,
                                                            max_scale_factor=1.0,
                                                            scale_factor_step_size=0,
                                                            is_training=data_augment)

        return image, labels

    dataset = dataset.map(parser, num_parallel_calls=num_parallel_calls)
    dataset = dataset.shuffle(buffer_size = buffer_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()

    return iterator
