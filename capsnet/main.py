import os  # nopep8
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8

from capsnet import flag
from capsnet.data import dataset
from capsnet.models import capsule_ori, network, pose, vggface, resnet50, capsule, capsule_mod, capsule_3d
from capsnet.utils import scheduler
import time
import pdb
import sys
import traceback
import tensorflow as tf
from absl import logging, flags, app
from tqdm import tqdm

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

FLAGS = flags.FLAGS


class LogMeter(object):
  def __init__(self, logs_dir):
    self.logs_dir = logs_dir
    self.log = dict()
    if logs_dir:
      self._writer = tf.summary.create_file_writer(logs_dir)
      self._writer.set_as_default()

  def update(self, log_update):
    for key in log_update:
      if key in self.log:
        self.log[key].append(log_update[key])
      else:
        self.log[key] = [log_update[key]]

  def average(self, step):
    for key in self.log:
      self.log[key] = tf.reduce_mean(self.log[key])
      if self.logs_dir:
        tf.summary.scalar(key, self.log[key], step=step)

  def reset(self):
    self.log.clear()

  def get_item(self, key):
    return self.log[key].numpy()


def main(argv):
  # 1. Create model: Feature extractor
  if FLAGS.type == 'pose_only':
    net = network.Network(
        model=pose.PoseEstimator({
            'pose_lambda_d': FLAGS.pose_lambda_d,
            'pose_stage_base': FLAGS.pose_stage_base,
            'pose_n_caps1': FLAGS.pose_n_caps1,
            'pose_n_caps1_dim': FLAGS.pose_n_caps1_dim,
            'pose_n_caps2': FLAGS.pose_n_caps2,
            'pose_n_caps2_dim': FLAGS.pose_n_caps2_dim,
            'pose_routing': FLAGS.pose_routing,
        }),
        checkpoint_dir=FLAGS.checkpoint_dir,
        optimizer_type=FLAGS.optimizer,
        learning_rate=scheduler.lr_func)
    loss_metrics = {
        'pose': [FLAGS.loss_pose, FLAGS.loss_pose_weight],
        'pose_orth': [FLAGS.loss_orth, FLAGS.loss_orth_weight],
    }
  elif FLAGS.type == 'siamese_ssr':
    net = network.Network(
        model=capsule.Extractor({
            'fe_n_caps1': FLAGS.fe_n_caps1,
            'fe_n_caps1_dim': FLAGS.fe_n_caps1_dim,
            'fe_n_caps2': FLAGS.fe_n_caps2,
            'fe_n_caps2_dim': FLAGS.fe_n_caps2_dim,
            'fe_routing': FLAGS.fe_routing,
        }),
        checkpoint_dir=FLAGS.checkpoint_dir,
        optimizer_type=FLAGS.optimizer,
        learning_rate=scheduler.lr_func)
    loss_metrics = {
        'fe_verif': [FLAGS.loss_fe_verif, FLAGS.loss_fe_verif_weight],
        'fe_iden': [FLAGS.loss_fe_iden, FLAGS.loss_fe_iden_weight],
    }
  elif FLAGS.type == 'siamese_ori':
    net = network.Network(
        model=capsule_ori.Extractor({
            'fe_n_caps1': FLAGS.fe_n_caps1,
            'fe_n_caps1_dim': FLAGS.fe_n_caps1_dim,
            'fe_n_caps2': 100,  # FLAGS.fe_n_caps2,
            'fe_n_caps2_dim': FLAGS.fe_n_caps2_dim,
            'fe_routing': FLAGS.fe_routing,
        }),
        checkpoint_dir=FLAGS.checkpoint_dir,
        optimizer_type=FLAGS.optimizer,
        learning_rate=scheduler.lr_func)
    loss_metrics = {
        'fe_iden': [FLAGS.loss_fe_iden, FLAGS.loss_fe_iden_weight],
    }
  elif FLAGS.type == 'siamese_mod':
    net = network.Network(
        model=capsule_mod.Extractor({
            'fe_n_caps1': FLAGS.fe_n_caps1,
            'fe_n_caps1_dim': FLAGS.fe_n_caps1_dim,
            'fe_n_caps2': FLAGS.fe_n_caps2,
            'fe_n_caps2_dim': FLAGS.fe_n_caps2_dim,
        }),
        checkpoint_dir=FLAGS.checkpoint_dir,
        optimizer_type=FLAGS.optimizer,
        learning_rate=scheduler.lr_func)
    loss_metrics = {
        'fe_verif': [FLAGS.loss_fe_verif, FLAGS.loss_fe_verif_weight],
        'fe_iden': [FLAGS.loss_fe_iden, FLAGS.loss_fe_iden_weight],
    }
  elif FLAGS.type == 'siamese_3d':
    net = network.Network(
        model=capsule_3d.Extractor({
            'fe_n_caps1': 8,
            'fe_n_caps1_dim': 32,
            'fe_n_caps2': 16,
            'fe_n_caps2_dim': 16,
            'fe_n_caps3': 10177,
            'fe_n_caps3_dim': 16,
        }),
        checkpoint_dir=FLAGS.checkpoint_dir,
        optimizer_type=FLAGS.optimizer,
        learning_rate=scheduler.lr_func)
    loss_metrics = {
        'fe_verif': [FLAGS.loss_fe_verif, FLAGS.loss_fe_verif_weight],
        'fe_iden': [FLAGS.loss_fe_iden, FLAGS.loss_fe_iden_weight],
    }
  elif FLAGS.type == 'resnet':
    net = network.Network(
        model=resnet50.Extractor({}),
        checkpoint_dir=FLAGS.checkpoint_dir,
        optimizer_type=FLAGS.optimizer,
        learning_rate=scheduler.lr_func)
    loss_metrics = {
        'fe_iden': [FLAGS.loss_fe_iden, FLAGS.loss_fe_iden_weight],
    }
  else:
    raise ValueError(f'Type {FLAGS.type} not recognized')

  # 2. Restore model if restart training
  if FLAGS.restart_training:
    print(f'Restore model from checkpoint {FLAGS.checkpoint_dir}')
    net.restore()

  # 3. Setup log & tensorboard
  log = LogMeter(FLAGS.logs_dir)

  # 4. Train / Test loop
  if FLAGS.mode == 'train':
    # 4.1. Make training iterator
    train_it = dataset.make_dataset_iterator(
        data_name=FLAGS.dataset,
        mode=FLAGS.mode,
        supervised=FLAGS.supervised,
        data_dir=FLAGS.data_dir,
        data_dim=FLAGS.data_dim,
        data_gen=FLAGS.data_gen,
        batch_size=FLAGS.batch_size,
        train_size=FLAGS.train_size,
        val_size=FLAGS.val_size,
        test_size=FLAGS.test_size,
    )

    epoch_length = train_it.cardinality().numpy()
    epoch_start = tf.compat.v1.train.get_or_create_global_step() // epoch_length

    # 4.2. Training loop
    print('\nStart training...\n')
    for current_epoch in range(epoch_start, FLAGS.n_epoch):
      print(f'Epoch {current_epoch+1} / {FLAGS.n_epoch}:')

      if FLAGS.prog_bar:
        pbar = tqdm(total=epoch_length)

      # ----------------------- Train 1 epoch -----------------------
      start_time_data = time.time()
      for batch in iter(train_it):
        stop_time_data = time.time()

        # ---------------------- Train 1 batch ----------------------
        start_time_train = time.time()
        log_update = net.train(
            batch,
            train_eager=FLAGS.train_eager,
            loss_metrics=loss_metrics,
        )
        stop_time_train = time.time()

        if FLAGS.prog_bar:
          pbar.set_postfix(
              loss_total='%.4f' % tf.reduce_mean(log_update['loss_total']).numpy())
          pbar.update()
        # -------------------- Train 1 batch end --------------------

        # ------------------------- Logging -------------------------
        log_update['data_time'] = (stop_time_data - start_time_data) * 1000
        log_update['train_time'] = (stop_time_train - start_time_train) * 1000
        log.update(log_update)
        start_time_data = time.time()
        # ----------------------- Logging end -----------------------
      # --------------------- Train 1 epoch end ---------------------

      log.average(current_epoch)
      print(f"\n\tloss_total: {log.get_item('loss_total'):.4f}\n")
      log.reset()

      if FLAGS.checkpoint_dir:
        net.save()

      # Run a validation loop at the end of each epoch
      # WIP

  elif FLAGS.mode == 'test':    # WIP
    pass
  else:
    raise ValueError(
        f'Mode {FLAGS.mode} not recognized')


if __name__ == '__main__':
  try:
    app.run(main)
  except Exception as err:
    last_traceback = sys.exc_info()[2]
    traceback.print_tb(last_traceback)
    print(err)
    # pdb.post_mortem(last_traceback)
