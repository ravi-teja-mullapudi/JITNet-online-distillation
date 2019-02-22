"""Train JITNet on the COCO dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.python.ops import control_flow_ops

sys.path.append(os.path.realpath('./datasets'))

import res_seg
import coco_tfrecords as coco

sys.path.append('./models/research/slim/deployment')
import model_deploy as model_deploy

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer(
    'num_clones', 1,
    'Number of model clones to deploy (number of GPU devices used).')

tf.app.flags.DEFINE_boolean(
    'clone_on_cpu', False,
    'Use the CPU to deploy clones.')

tf.app.flags.DEFINE_integer(
    'worker_replicas', 1, 
    'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'Number of parameter servers. If 0, then parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'Frequency with which logs are printed.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 120,
    'Frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 300,
    'Frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'startup_delay_steps', 15,
    'Number of training steps between replica startups.')

tf.app.flags.DEFINE_integer(
    'task', 0,
    'Task ID of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004,
    'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'momentum',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float(
    'opt_epsilon', 1.0,
    'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float(
    'ftrl_learning_rate_power', -0.5,
    'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0,
    'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0,
    'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float(
    'rmsprop_decay', 0.9,
    'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type', 'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float(
    'learning_rate', 0.001,
    'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.00001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.1,
    'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.1,
    'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 25.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', 0.9999,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'train_data_file', None,
    'The name of the train/test split.')

tf.app.flags.DEFINE_integer(
    'height', 1024,
    'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'width', 2048,
    'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32,
    'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_number_of_steps', None,
    'The maximum number of training steps.')

tf.app.flags.DEFINE_integer(
    'num_samples_per_epoch', 4000,
    'Number of samples per epoch.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

tf.app.flags.DEFINE_boolean(
    'use_seperable_convolution', False,
    'Use a seperable convolution block.')

tf.app.flags.DEFINE_float(
    'filter_depth_multiplier', 1.0,
    'Filter depth multipler for encoder.')

tf.app.flags.DEFINE_integer(
    'num_units', 1,
    'Number of units in each ressep block.')

tf.app.flags.DEFINE_float(
    'scale', 1.0, 'Input scale factor')

tf.app.flags.DEFINE_integer(
    'foreground_weight', 10,
    'Weights for foreground objects.')

tf.app.flags.DEFINE_integer(
    'background_weight', 10,
    'Weights for background objects.')

tf.app.flags.DEFINE_boolean(
    'use_batch_norm', True,
    'Use batch normalization.')

FLAGS = tf.app.flags.FLAGS

def _configure_learning_rate(num_samples_per_epoch, global_step, num_clones):
  """Configure the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if learning rate decay type is not recognized.
  """
  decay_steps = int(num_samples_per_epoch / (FLAGS.batch_size * num_clones) *
                                             FLAGS.num_epochs_per_decay)
  if FLAGS.sync_replicas:
    decay_steps /= FLAGS.replicas_to_aggregate

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(FLAGS.learning_rate,
                                     global_step,
                                     decay_steps,
                                     FLAGS.end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     FLAGS.learning_rate_decay_type)

def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.

    Args:
        learning_rate: A scalar or `Tensor` learning rate.

    Returns:
        An instance of an optimizer.

    Raises:
        ValueError: if FLAGS.optimizer is not recognized.
    """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=FLAGS.adadelta_rho,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=FLAGS.ftrl_learning_rate_power,
            initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
            l1_regularization_strength=FLAGS.ftrl_l1,
            l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=FLAGS.momentum,
            name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.momentum,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer

def _get_init_fn():
    """Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
        An init function run by the supervisor.
    """
    if FLAGS.checkpoint_path is None:
        return None

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(FLAGS.train_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % FLAGS.train_dir)
        return None

    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                     for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=FLAGS.ignore_missing_vars)

def _get_variables_to_train():
    """Returns a list of variables to train.

    Returns:
        A list of variables to train by the optimizer.
    """
    if FLAGS.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train

def resseg_model(input, height, width, scale, weight_decay,
                 use_seperable_convolution, num_classes,
                 filter_depth_multiplier=1, num_units=1,
                 is_training=True, use_batch_norm=True):
    model = res_seg
    with slim.arg_scope(model.ressep_arg_scope(weight_decay=weight_decay,
                                               use_batch_norm=use_batch_norm)):
        net, end_points = model.ressep_factory(
                                input,
                                use_seperable_convolution = use_seperable_convolution,
                                filter_depth_multiplier = filter_depth_multiplier,
                                is_training = is_training,
                                use_batch_norm = use_batch_norm,
                                num_units = num_units,
                                num_classes = num_classes,
                                scale = scale)

        return net, end_points

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    #######################
    # Config model_deploy #
    #######################
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=FLAGS.num_clones,
        clone_on_cpu=FLAGS.clone_on_cpu,
        replica_id=FLAGS.task,
        num_replicas=FLAGS.worker_replicas,
        num_ps_tasks=FLAGS.num_ps_tasks)

    # Create global_step
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    with tf.device(deploy_config.inputs_device()):
        iterator = coco.get_dataset(FLAGS.train_data_file,
                                    batch_size = FLAGS.batch_size,
                                    num_epochs = 500,
                                    buffer_size = 250 * FLAGS.num_clones,
                                    num_parallel_calls = 4 * FLAGS.num_clones,
                                    crop_height = FLAGS.height,
                                    crop_width = FLAGS.width,
                                    resize_shape = FLAGS.width,
                                    data_augment = True)

    def clone_fn(iterator):
        with tf.device(deploy_config.inputs_device()):
            batch_image, batch_labels = iterator.get_next()

        s = batch_labels.get_shape().as_list()
        batch_labels.set_shape([FLAGS.batch_size, s[1], s[2], s[3]])

        s = batch_image.get_shape().as_list()
        batch_image.set_shape([FLAGS.batch_size, s[1], s[2], s[3]])

        num_classes = coco.num_classes()

        logits, end_points = resseg_model(batch_image, FLAGS.height,
                                          FLAGS.width, FLAGS.scale,
                                          FLAGS.weight_decay,
                                          FLAGS.use_seperable_convolution,
                                          num_classes,
                                          is_training=True,
                                          use_batch_norm=FLAGS.use_batch_norm,
                                          num_units=FLAGS.num_units,
                                          filter_depth_multiplier=FLAGS.filter_depth_multiplier)

        s = logits.get_shape().as_list()
        with tf.device(deploy_config.inputs_device()):
            lmap_size = 256
            lmap = np.array([0]* lmap_size)
            for k, v in coco.id2trainid_objects.items():
                lmap[k] = v + 1
            lmap = tf.constant(lmap, tf.uint8)
            down_labels = tf.cast(batch_labels, tf.int32)
            label_mask  = tf.squeeze((down_labels < 255))
            down_labels = tf.gather(lmap, down_labels)
            down_labels = tf.cast(down_labels, tf.int32)
            down_labels = tf.reshape(down_labels, tf.TensorShape([FLAGS.batch_size, s[1], s[2]]))
            down_labels = tf.cast(label_mask, tf.int32) * down_labels

            fg_weights = tf.constant(FLAGS.foreground_weight,
                                     dtype=tf.int32, shape=label_mask.shape)
            label_weights = tf.cast(label_mask, tf.int32) * fg_weights

        # Specify the loss
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(down_labels,
                                                               logits,
                                                               weights = label_weights,
                                                               scope='xentropy')
        tf.losses.add_loss(cross_entropy)

        return end_points, batch_image, down_labels, logits

    # Gather initial summaries
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    clones = model_deploy.create_clones(deploy_config, clone_fn, [iterator])
    first_clone_scope = deploy_config.clone_scope(0)
    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by network_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    #################################
    # Configure the moving averages #
    #################################
    if FLAGS.moving_average_decay:
        moving_average_variables = slim.get_model_variables()
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay, global_step)
    else:
        moving_average_variables, variable_averages = None, None

    #########################################
    # Configure the optimization procedure. #
    #########################################
    with tf.device(deploy_config.optimizer_device()):
        learning_rate = _configure_learning_rate(FLAGS.num_samples_per_epoch,
                                               global_step, deploy_config.num_clones)
        optimizer = _configure_optimizer(learning_rate)
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    if FLAGS.sync_replicas:
        # If sync_replicas is enabled, the averaging will be done in the chief
        # queue runner.
        optimizer = tf.train.SyncReplicasOptimizer(
            opt=optimizer,
            replicas_to_aggregate=FLAGS.replicas_to_aggregate,
            variable_averages=variable_averages,
            variables_to_average=moving_average_variables,
            replica_id=tf.constant(FLAGS.task, tf.int32, shape=()),
            total_num_replicas=FLAGS.worker_replicas)
    elif FLAGS.moving_average_decay:
        # Update ops executed locally by trainer.
        update_ops.append(variable_averages.apply(moving_average_variables))

    end_points, batch_image, down_labels, logits = clones[0].outputs

    cmap = np.array(coco.id2color)
    cmap = tf.constant(cmap, tf.uint8)
    seg_map = tf.gather(cmap, down_labels)

    predictions = tf.argmax(logits, axis=3)
    pred_map = tf.gather(cmap, predictions)

    summaries.add(tf.summary.image('labels', seg_map))
    summaries.add(tf.summary.image('predictions', pred_map))
    summaries.add(tf.summary.image('images', batch_image))

    # Variables to train.
    variables_to_train = _get_variables_to_train()

    # Returns a train_tensor and summary_op
    total_loss, clones_gradients = model_deploy.optimize_clones(clones,
        optimizer,
        var_list=variables_to_train)
    
    # Add total_loss to summary.
    summaries.add(tf.summary.scalar('total_loss', total_loss))

    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(clones_gradients,
                                             global_step=global_step)
    update_ops.append(grad_updates)

    update_op = tf.group(*update_ops)
    train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
                                                      name='train_op')

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))
    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    if FLAGS.sync_replicas:
        sync_optimizer = opt
        startup_delay_steps = 0
    else:
        sync_optimizer = None
        startup_delay_steps = FLAGS.task * FLAGS.startup_delay_steps

    ###########################
    # Kick off the training.  #
    ###########################
    slim.learning.train(
        train_tensor,
        logdir=FLAGS.train_dir,
        master=FLAGS.master,
        is_chief=(FLAGS.task == 0),
        init_fn=_get_init_fn(),
        summary_op=summary_op,
        number_of_steps=FLAGS.max_number_of_steps,
        log_every_n_steps=FLAGS.log_every_n_steps,
        startup_delay_steps=startup_delay_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        sync_optimizer=sync_optimizer)

if __name__ == '__main__':
    tf.app.run()
