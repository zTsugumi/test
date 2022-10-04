from absl import flags
import tensorflow as tf
from capsnet import flag

FLAGS = flags.FLAGS


def lr_func():
  step = tf.compat.v1.train.get_or_create_global_step()
  effective_step = tf.maximum(step - FLAGS.lr_decay_after_step + 1, 0)

  lr_step_ratio = tf.cast(effective_step, dtype=tf.float32) / float(FLAGS.lr_decay_steps)

  if FLAGS.lr_decay_type == 'none' or FLAGS.lr_decay_steps <= 0:
    return FLAGS.lr
  elif FLAGS.lr_decay_type == 'linear':
    return FLAGS.lr * tf.maximum(1.0 - lr_step_ratio, 0.0)
  elif FLAGS.lr_decay_type == 'exponential':
    return FLAGS.lr * FLAGS.lr_decay_rate**lr_step_ratio
  else:
    raise ValueError(f'Decay type {FLAGS.lr_decay_type} not found')
