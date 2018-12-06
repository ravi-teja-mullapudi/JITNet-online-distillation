from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import resnet_utils

slim = tf.contrib.slim

ressep_arg_scope = resnet_utils.ressep_arg_scope
separable_conv2d = slim.separable_conv2d

@slim.add_arg_scope
def basic(inputs, depth, stride, rate=1, use_batch_norm=True,
          outputs_collections=None, scope=None):
  with tf.variable_scope(scope, 'basic_v2', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    if use_batch_norm:
      preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
    else:
      preact = tf.contrib.layers.layer_norm(inputs, activation_fn=tf.nn.relu, scope='preact')

    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                             normalizer_fn=None, activation_fn=None,
                             scope='shortcut')

    residual = resnet_utils.conv2d_same(preact, depth, 3, stride,
                                        rate=rate, scope='conv1')

    residual = slim.conv2d(residual, depth, [1, 3], stride=1,
                           normalizer_fn=None, activation_fn=None,
                           scope='conv2_1x3')

    residual = slim.conv2d(residual, depth, [3, 1], stride=1,
                           normalizer_fn=None, activation_fn=None,
                           scope='conv2_3x1')

    output = shortcut + residual

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.original_name_scope,
                                            output)

def ressep_backseg(inputs,
                   frame_encoder_filter_sizes = [16, 64, 128, 256, 512],
                   background_decoder_filter_sizes = [128, 64, 32, 16, 8],
                   use_seperable_convolution = False,
                   num_classes = None,
                   is_training = True,
                   use_batch_norm = True,
                   freeze_batch_norm = False,
                   num_units = 1,
                   depth_multiplier = 1,
                   filter_depth_multiplier = 1.0,
                   min_enc_filters = 8,
                   min_dec_filters = 16,
                   reuse = None,
                   scope = None,
                   scale = 1.0):

    background_decoder_num_units = [ num_units for _ in background_decoder_filter_sizes ]
    frame_encoder_num_units = [ num_units for _ in frame_encoder_filter_sizes ]

    assert(len(frame_encoder_filter_sizes) == len(frame_encoder_num_units))
    assert(len(background_decoder_filter_sizes) == len(background_decoder_num_units))

    assert(len(background_decoder_filter_sizes) == len(frame_encoder_filter_sizes))

    in_shape = inputs.shape.as_list()
    h = in_shape[1]
    w = in_shape[2]

    if (scale < 1.0):
        original_dims = [h, w]
        rescale_dims = [int(scale * h), int(scale * w)]
        low_res_inputs = tf.image.resize_images(inputs, rescale_dims)
    else:
        low_res_inputs = inputs

    with tf.variable_scope(scope, 'ressep_dual', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d], outputs_collections=end_points_collection):
            train_bnorm = is_training and not freeze_batch_norm
            with slim.arg_scope([slim.batch_norm], is_training=train_bnorm):
                with tf.variable_scope('path1'):
                    net = low_res_inputs
                    shp = net.get_shape().as_list()
                    conv1_in_size = [shp[1], shp[2]]
                    channels = max(int(16 * filter_depth_multiplier), min_enc_filters)
                    net = resnet_utils.conv2d_same(net, channels, 3, stride=2, scope='conv1')
                    shp = net.get_shape().as_list()
                    conv2_in_size = [shp[1], shp[2]]
                    channels = max(int(64 * filter_depth_multiplier), min_enc_filters)
                    net = resnet_utils.conv2d_same(net, channels, 3, stride=2, scope='conv2')
                    frame_encoder_outs = []
                    frame_encoder_sizes = []
                    with tf.variable_scope('frame_encoder', values = [net]):
                        for b in range(len(frame_encoder_filter_sizes)):
                            shp = net.get_shape().as_list()
                            frame_encoder_sizes.append([shp[1], shp[2]])

                            filter_depth = frame_encoder_filter_sizes[b]
                            num_units = frame_encoder_num_units[b]
                            stride = 1
                            for u in range(num_units - 1):
                                if use_seperable_convolution:
                                    net = basic_sep(net, filter_depth, stride, use_batch_norm=use_batch_norm)
                                else:
                                    net = basic(net, filter_depth, stride, use_batch_norm=use_batch_norm)
                            # Downsample
                            stride = 2
                            if use_seperable_convolution:
                                net = basic_sep(net, filter_depth, stride, use_batch_norm=use_batch_norm)
                            else:
                                net = basic(net, filter_depth, stride, use_batch_norm=use_batch_norm)

                            frame_encoder_outs.append(net)

                    # background foreground decoder
                    with tf.variable_scope('background_decoder', values = [net]):
                        num_decoder_blocks = len(background_decoder_filter_sizes)
                        net = None

                        for b in range(num_decoder_blocks):
                            filter_depth = background_decoder_filter_sizes[b]
                            num_units = background_decoder_num_units[b]
                            stride = 1

                            frame_stream = frame_encoder_outs[num_decoder_blocks - b - 1]
                            if net is None:
                                net = frame_stream
                            else:
                                net = tf.concat([frame_stream, net], axis=3)

                            for u in range(num_units):
                                if use_seperable_convolution:
                                    net = basic_sep(net, filter_depth, stride, use_batch_norm=use_batch_norm)
                                else:
                                    net = basic(net, filter_depth, stride, use_batch_norm=use_batch_norm)

                            stream_size = frame_encoder_sizes[num_decoder_blocks - 1 - b]

                            # Usample features
                            net = tf.image.resize_images(net, (stream_size), align_corners=True)

                        net = tf.image.resize_images(net, (conv2_in_size), align_corners=True)
                        channels = max(int(64 * filter_depth_multiplier), min_dec_filters)
                        net = resnet_utils.conv2d_same(net, channels, 3, stride=1, scope='decoder_conv2')
                        channels = max(int(16 * filter_depth_multiplier), min_dec_filters)
                        net = resnet_utils.conv2d_same(net, channels, 3, stride=1, scope='decoder_conv1')

                        net = tf.image.resize_images(net, (conv1_in_size), align_corners=True)
                        path1_end = net

                with tf.variable_scope('path2'):
                    net = low_res_inputs
                    shp = net.get_shape().as_list()
                    conv1_in_size = [shp[1], shp[2]]
                    channels = max(int(16 * filter_depth_multiplier), min_enc_filters)
                    net = resnet_utils.conv2d_same(net, channels, 3, stride=2, scope='conv1')
                    shp = net.get_shape().as_list()
                    conv2_in_size = [shp[1], shp[2]]
                    channels = max(int(64 * filter_depth_multiplier), min_enc_filters)
                    net = resnet_utils.conv2d_same(net, channels, 3, stride=2, scope='conv2')
                    frame_encoder_outs = []
                    frame_encoder_sizes = []
                    with tf.variable_scope('frame_encoder', values = [net]):
                        for b in range(len(frame_encoder_filter_sizes)):
                            shp = net.get_shape().as_list()
                            frame_encoder_sizes.append([shp[1], shp[2]])

                            filter_depth = frame_encoder_filter_sizes[b]
                            num_units = frame_encoder_num_units[b]
                            stride = 1
                            for u in range(num_units - 1):
                                if use_seperable_convolution:
                                    net = basic_sep(net, filter_depth, stride, use_batch_norm=use_batch_norm)
                                else:
                                    net = basic(net, filter_depth, stride, use_batch_norm=use_batch_norm)
                            # Downsample
                            stride = 2
                            if use_seperable_convolution:
                                net = basic_sep(net, filter_depth, stride, use_batch_norm=use_batch_norm)
                            else:
                                net = basic(net, filter_depth, stride, use_batch_norm=use_batch_norm)

                            frame_encoder_outs.append(net)

                    # background foreground decoder
                    with tf.variable_scope('background_decoder', values = [net]):
                        num_decoder_blocks = len(background_decoder_filter_sizes)
                        net = None

                        for b in range(num_decoder_blocks):
                            filter_depth = background_decoder_filter_sizes[b]
                            num_units = background_decoder_num_units[b]
                            stride = 1

                            frame_stream = frame_encoder_outs[num_decoder_blocks - b - 1]
                            if net is None:
                                net = frame_stream
                            else:
                                net = tf.concat([frame_stream, net], axis=3)

                            for u in range(num_units):
                                if use_seperable_convolution:
                                    net = basic_sep(net, filter_depth, stride, use_batch_norm=use_batch_norm)
                                else:
                                    net = basic(net, filter_depth, stride, use_batch_norm=use_batch_norm)

                            stream_size = frame_encoder_sizes[num_decoder_blocks - 1 - b]

                            # Usample features
                            net = tf.image.resize_images(net, (stream_size), align_corners=True)

                        net = tf.image.resize_images(net, (conv2_in_size), align_corners=True)
                        channels = max(int(64 * filter_depth_multiplier), min_dec_filters)
                        net = resnet_utils.conv2d_same(net, channels, 3, stride=1, scope='decoder_conv2')
                        channels = max(int(16 * filter_depth_multiplier), min_dec_filters)
                        net = resnet_utils.conv2d_same(net, channels, 3, stride=1, scope='decoder_conv1')

                        net = tf.image.resize_images(net, (conv1_in_size), align_corners=True)
                        path2_end = net

        net = tf.concat([path1_end, path2_end], axis=3)

        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)

        if scale < 1.0:
            net = tf.image.resize_images(net, (in_shape[1], in_shape[2]),
                                         align_corners=True)

        end_points['final_full_res'] = net
        end_points['low_res_inputs'] = low_res_inputs

        logits = slim.conv2d(net, num_classes, [1, 1], normalizer_fn=None,
                             activation_fn=None, scope='logits')
        return logits, end_points

def ressep_factory(inputs,
                   filter_depth_multiplier = 1,
                   num_units = 1,
                   use_seperable_convolution = False,
                   is_training=True,
                   reuse=None,
                   num_classes=None,
                   use_batch_norm=True,
                   freeze_batch_norm=False,
                   scale = 1.0,
                   scope='ressep'):

    frame_encoder_filter_sizes = [128, 128, 256]
    background_decoder_filter_sizes = [256, 128, 64]

    min_dec_filters = 8
    if num_classes > 8:
        min_dec_filters = 16
    if num_classes > 16:
        min_dec_filters = 32

    min_enc_filters = 8

    background_decoder_filter_sizes = [ max(int(filter_depth_multiplier * s), min_dec_filters) \
                                        for s in background_decoder_filter_sizes ]
    frame_encoder_filter_sizes = [ max(int(filter_depth_multiplier * s), min_enc_filters) \
                                   for s in frame_encoder_filter_sizes ]

    return ressep_backseg(inputs,
                          frame_encoder_filter_sizes = frame_encoder_filter_sizes,
                          background_decoder_filter_sizes = background_decoder_filter_sizes,
                          use_seperable_convolution = use_seperable_convolution,
                          is_training = is_training,
                          scope = scope,
                          num_classes = num_classes,
                          use_batch_norm = use_batch_norm,
                          freeze_batch_norm = freeze_batch_norm,
                          num_units = num_units,
                          filter_depth_multiplier = filter_depth_multiplier,
                          min_enc_filters = min_enc_filters,
                          min_dec_filters = min_dec_filters,
                          scale = scale)
