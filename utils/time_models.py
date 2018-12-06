from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
import math

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import slim

sys.path.append(os.path.realpath('./src'))
import res_seg

def model_builder(model_fn,
                  scale = 1.0,
                  batch_size = 1,
                  height = 1080,
                  width = 1920,
                  channels = 3,
                  filter_depth_multiplier=1,
                  num_units = 1):

    batch_image = tf.random_normal([batch_size, height, width, channels])
    return model_fn(tf.cast(batch_image, tf.float32),
                    height, width, scale, 0.0, False, 23,
                    filter_depth_multiplier = filter_depth_multiplier,
                    num_units = num_units)

def time_model(model_fn,
               model_builder,
               end_point,
               scale = 1.0,
               num_iterations = 50,
               batch_size = 1,
               height = 1024,
               width = 2048,
               channels = 3,
               filter_depth_multiplier = 1,
               num_units = 1,
               profile = False):

    tf.reset_default_graph()
    min_time = float('inf')
    with tf.Graph().as_default():
        net, end_points = model_builder(model_fn,
                                      scale = scale,
                                      batch_size = batch_size,
                                      height = height,
                                      width = width,
                                      channels = channels,
                                      filter_depth_multiplier = filter_depth_multiplier,
                                      num_units = num_units)

        predictions = tf.argmax(net, axis=3)
        end_points['predictions'] = predictions

        if profile:
            prof = tf.profiler.profile(tf.get_default_graph(),
                                options=tf.profiler.ProfileOptionBuilder.float_operation())
            print('Total flops:', prof.total_float_ops)
            print(np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))
        # allow growth so that large memory allocations are cut down
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction=0.5

        with tf.Session(config=config) as sess:
            run_metadata = tf.RunMetadata()
            sess.run(tf.global_variables_initializer())
            for i in range(num_iterations):
                start = time.time()
                vals = sess.run(end_points[end_point])
                end = time.time()
                min_time = min(min_time, end - start)
    return min_time

def ressep_model(input, height, width, scale, weight_decay,
                 use_seperable_convolution, num_classes,
                 filter_depth_multiplier=1, num_units=1,
                 is_training=True, use_batch_norm=True):
    
    with slim.arg_scope(res_seg.ressep_arg_scope(weight_decay=weight_decay,
                                                       use_batch_norm=use_batch_norm)):
        net, end_points = res_seg.ressep_factory(
                                input,
                                use_seperable_convolution = use_seperable_convolution,
                                filter_depth_multiplier = filter_depth_multiplier,
                                is_training = is_training,
                                use_batch_norm = use_batch_norm,
                                num_units = num_units,
                                num_classes = num_classes,
                                scale = scale)

        return net, end_points

print('JITNet:', time_model(ressep_model,
                               model_builder,
                               'predictions',
                               num_iterations = 10,
                               height = 720,
                               width = 1280,
                               scale = 1.0,
                               filter_depth_multiplier = 0.5,
                               num_units = 1,
                               profile = True))
