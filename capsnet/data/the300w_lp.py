import os
import numpy as np
import tensorflow as tf
from capsnet.utils import ops_geo


def load_data_npz(path):
  d = np.load(path)
  poses = d['pose']
  vecs = []

  for pose in poses:
    pitch_rad, roll_rad, yaw_rad = pose * np.pi / 180.0
    temp_l_vec, temp_b_vec, temp_f_vec = ops_geo.EulerAngles2Vectors(roll_rad, pitch_rad, yaw_rad)
    vecs.append([temp_l_vec[0], temp_l_vec[1], temp_l_vec[2],
                 temp_b_vec[0], temp_b_vec[1], temp_b_vec[2],
                 temp_f_vec[0], temp_f_vec[1], temp_f_vec[2]])

  return d['image'], np.array(vecs)


def make_dataset(dir, mode, batch_size, data_split):
  data_list = ['AFW.npz']  # , 'AFW_Flip.npz', 'HELEN.npz', 'HELEN_Flip.npz',
  # 'IBUG.npz', 'IBUG_Flip.npz', 'LFPW.npz', 'LFPW_Flip.npz']
  images = []
  poses = []
  for data_name in data_list:
    data_dir = os.join(dir, data_name)
    image, pose = load_data_npz(data_dir)
    images.append(image)
    poses.append(pose)
  images = np.concatenate(images, 0)
  poses = np.concatenate(poses, 0)

  # Standardize data, remove angle outside of range [-99, 99]
  indices = np.all([poses.max(axis=1) <= 99.0, poses.min(axis=1) >= -99.0], axis=0)
  x_data = np.float32(images[indices, :, :, :]) / 255.0
  y_data = np.float32(poses[indices, :])

  dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

  data_size = dataset.cardinality().numpy()
  train_size, val_size, test_size = data_split
  train_size = int(data_size * train_size)
  val_size = int(data_size * val_size)
  test_size = int(data_size * test_size)

  if mode == 'train':
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)

    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(1)
    train_dataset = train_dataset.shuffle(1000)
    # train_dataset = train_dataset.repeat()

    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
    val_dataset = val_dataset.prefetch(1)

    # return iter(train_dataset), iter(val_dataset)
    return train_dataset, val_dataset
  elif mode == 'test':
    test_dataset = dataset.skip(train_size + val_size)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
    test_dataset = test_dataset.prefetch(1)

    # return iter(test_dataset)
    return test_dataset
  else:
    raise ValueError(f'Data mode {mode} not recognized')
