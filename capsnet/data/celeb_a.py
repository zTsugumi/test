import os  # nopep8
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8

import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tqdm import tqdm
from typing import *

AUTOTUNE = tf.data.experimental.AUTOTUNE
FILE_COLUMN = 0
PERSON_COLUMN = 1
ID_COLUMN = 2
RANDOM_STATE = 1


def load_image(image_dir, target_dim):
  image = tf.image.decode_jpeg(tf.io.read_file(image_dir), channels=3)
  # image = tf.keras.applications.imagenet_utils.preprocess_input(image, mode='tf')
  image = tf.image.resize_with_crop_or_pad(image, 178, 178)
  image = tf.image.resize(image, [target_dim, target_dim])
  image = tf.keras.applications.resnet50.preprocess_input(image)

  return image


def build_partitions(
        identity_df: pd.DataFrame,
        partition_dir: str,
        partition_size: Tuple[float, float, float]) -> str:
  '''
  Builds dataset partitions and return path to the info file
  '''
  train_size, val_size, test_size = partition_size

  print(f'Generating partitions (Train={train_size}, Val={val_size}, Test={test_size})')
  ids = np.unique(identity_df[PERSON_COLUMN])
  np.random.shuffle(ids)
  ids = np.random.choice(ids, 1000, replace=False)

  id_size = ids.shape[0]
  train_size = 100  # int(id_size * train_size)
  val_size = int(id_size * val_size)
  test_size = int(id_size * test_size)

  train_ids = ids[:train_size]
  val_ids = ids[train_size:train_size+val_size]
  test_ids = ids[train_size+val_size:]

  os.makedirs(partition_dir, exist_ok=True)

  json_file = os.path.join(partition_dir, 'partitions.json')
  if not os.path.exists(json_file):
    with open(json_file, 'w') as f:
      json.dump({
          'train': [int(x) for x in train_ids],
          'val': [int(x) for x in val_ids],
          'test': [int(x) for x in test_ids]}, f, indent=4)
  else:
    print('Partitions file already generated')

  return json_file


def get_pairs(identity_df: pd.DataFrame, ids: Iterable) -> Tuple[pd.DataFrame, pd.DataFrame]:
  def get_genuine_pairs() -> pd.DataFrame:
    print('Generating genuine pairs')
    paired_rows = pd.DataFrame(columns=['file_a', 'person_a', 'file_b', 'person_b'])
    for id in tqdm(ids):
      my_rows = identity_df[identity_df[PERSON_COLUMN] == id]
      my_rows_a = pd.DataFrame({'file_a': my_rows[FILE_COLUMN], 'person_a': my_rows[PERSON_COLUMN]})
      my_rows_b = pd.DataFrame({'file_b': my_rows[FILE_COLUMN], 'person_b': my_rows[PERSON_COLUMN]})
      my_paired_rows = my_rows_a.merge(my_rows_b, how='cross')
      my_paired_rows = my_paired_rows[my_paired_rows['file_a'] != my_paired_rows['file_b']]
      if my_paired_rows.shape[0] > my_rows.shape[0]:
        # ensures that the generated dataset has no more rows than the original dataset
        my_paired_rows = my_paired_rows.sample(n=my_rows.shape[0], random_state=RANDOM_STATE)
      paired_rows = paired_rows.append(my_paired_rows, ignore_index=True)
    return paired_rows

  def get_impostor_pairs() -> pd.DataFrame:
    print('Generating impostor pairs')
    paired_rows = pd.DataFrame(columns=['file_a', 'person_a', 'file_b', 'person_b'])
    for id in tqdm(ids):
      my_rows = identity_df[identity_df[PERSON_COLUMN] == id]
      # ensures that the generated dataset has no more rows than the original dataset
      other_rows = identity_df[identity_df[PERSON_COLUMN] != id].sample(n=my_rows.shape[0], random_state=RANDOM_STATE)
      my_rows_a = pd.DataFrame({'file_a': my_rows[FILE_COLUMN], 'person_a': my_rows[PERSON_COLUMN]})
      my_rows_b = pd.DataFrame({'file_b': other_rows[FILE_COLUMN], 'person_b': other_rows[PERSON_COLUMN]})
      my_paired_rows = my_rows_a.merge(my_rows_b, how='cross')
      if my_paired_rows.shape[0] > my_rows.shape[0]:
        # ensures that the generated dataset has no more rows than the original dataset
        my_paired_rows = my_paired_rows.sample(n=my_rows.shape[0], random_state=RANDOM_STATE)
      paired_rows = paired_rows.append(my_paired_rows, ignore_index=True)
    return paired_rows

  return get_genuine_pairs(), get_impostor_pairs()


def build_pairs(
        modes,
        identity_df: pd.DataFrame,
        partition_file: str,
        pair_dir: str) -> None:
  '''
  Builds pairs and return list of paths to info file
  '''
  with open(partition_file, 'r') as f:
    partitions = json.load(f)

  os.makedirs(pair_dir, exist_ok=True)

  pairs = {}
  for mode in modes:
    print(mode.upper() + ' PAIRS')
    genuine_file = os.path.join(pair_dir, f'{mode}_genuine_pairs.csv')
    impostor_file = os.path.join(pair_dir, f'{mode}_impostor_pairs.csv')
    pairs[mode] = (genuine_file, impostor_file)
    if not (os.path.exists(genuine_file) and os.path.exists(impostor_file)):
      genuine_pairs, impostor_pairs = get_pairs(identity_df, partitions[mode])
      genuine_pairs.to_csv(genuine_file)
      impostor_pairs.to_csv(impostor_file)
    else:
      print('Already generated')

  return pairs


# def preprocess(image_dir, genuine, impostor, target_dim):
#   genuine_a = image_dir + '/' + genuine['file_a']
#   genuine_a = load_image(genuine_a, target_dim)

#   genuine_b = image_dir + '/' + genuine['file_b']
#   genuine_b = load_image(genuine_b, target_dim)

#   impostor_a = image_dir + '/' + impostor['file_a']
#   impostor_a = load_image(impostor_a, target_dim)

#   impostor_b = image_dir + '/' + impostor['file_b']
#   impostor_b = load_image(impostor_b, target_dim)

#   pair_a = tf.stack([genuine_a, impostor_a], axis=0)
#   pair_b = tf.stack([genuine_b, impostor_b], axis=0)
#   gp = tf.stack([genuine['person_a'], genuine['person_b'], 1], axis=0)
#   ip = tf.stack([impostor['person_a'], impostor['person_b'], 0], axis=0)
#   pair_y = tf.stack([gp, ip], axis=0)
#   pair_y = tf.cast(pair_y, dtype=tf.float32)

#   return (pair_a, pair_b), pair_y


# def make_dataset(dir, data_dim, mode, batch_size, data_split):
#   identity_file = os.path.join(dir, 'identity_CelebA.txt')
#   identity_df = pd.read_csv(identity_file, sep=' ', header=None)

#   # 1. Build partitions
#   partition_dir = os.path.join(dir, 'celeba_partitions')
#   partition_file = build_partitions(identity_df, partition_dir, data_split)
#   print()

#   # 2. Build pairs & build dataset
#   pair_dir = os.path.join(dir, 'celeba_pairs')
#   image_dir = os.path.join(dir, 'img_align_celeba')
#   if mode == 'train':
#     pair_file = build_pairs(['train', 'val'], identity_df, partition_file, pair_dir)
#     train_genuine = pd.read_csv(pair_file['train'][0])
#     train_impostor = pd.read_csv(pair_file['train'][1])

#     # Sample the same amount from both genuine and impostor pair
#     n_sample = min(train_genuine.shape[0], train_impostor.shape[0])  # // 2
#     train_genuine = train_genuine.sample(n=n_sample, random_state=RANDOM_STATE)
#     train_impostor = train_impostor.sample(n=n_sample, random_state=RANDOM_STATE)

#     train_dataset = tf.data.Dataset.from_tensor_slices(
#         (dict(train_genuine), dict(train_impostor)))
#     train_dataset = train_dataset.map(
#         lambda genuine, impostor: preprocess(image_dir, genuine, impostor, data_dim),
#         num_parallel_calls=AUTOTUNE)

#     val_genuine = pd.read_csv(pair_file['val'][0])
#     val_impostor = pd.read_csv(pair_file['val'][1])

#     # Sample the same amount from both genuine and impostor pair
#     n_sample = min(val_genuine.shape[0], val_impostor.shape[0])  # // 2
#     val_genuine = val_genuine.sample(n=n_sample, random_state=RANDOM_STATE)
#     val_impostor = val_impostor.sample(n=n_sample, random_state=RANDOM_STATE)

#     val_dataset = tf.data.Dataset.from_tensor_slices(
#         (dict(val_genuine), dict(val_impostor)))
#     val_dataset = val_dataset.map(
#         lambda genuine, impostor: preprocess(image_dir, genuine, impostor, data_dim),
#         num_parallel_calls=AUTOTUNE)

#     train_dataset = train_dataset.unbatch()
#     train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
#     train_dataset = train_dataset.prefetch(1)
#     train_dataset = train_dataset.shuffle(1000)
#     # train_dataset = train_dataset.repeat()

#     val_dataset = val_dataset.unbatch()
#     val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
#     val_dataset = val_dataset.prefetch(1)

#     return train_dataset, val_dataset
#   elif mode == 'test':
#     pair_file = build_pairs(['test'], identity_df, partition_file, pair_dir)
#     test_genuine = pd.read_csv(pair_file['test'][0])
#     test_impostor = pd.read_csv(pair_file['test'][1])

#     # Sample the same amount from both genuine and impostor pair
#     n_sample = min(test_genuine.shape[0], test_impostor.shape[0])  # // 2
#     test_genuine = test_genuine.sample(n=n_sample, random_state=RANDOM_STATE)
#     test_impostor = test_impostor.sample(n=n_sample, random_state=RANDOM_STATE)

#     test_dataset = tf.data.Dataset.from_tensor_slices(
#         (dict(test_genuine), dict(test_impostor)))
#     test_dataset = test_dataset.map(
#         lambda genuine, impostor: preprocess(image_dir, genuine, impostor, data_dim),
#         num_parallel_calls=AUTOTUNE)

#     test_dataset = test_dataset.unbatch()
#     test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
#     test_dataset = test_dataset.prefetch(1)

#     return test_dataset
#   else:
#     raise ValueError(f'Data mode {mode} not recognized')


# Classification set
def preprocess(data, image_dir, target_dim):
  image_dir = image_dir + '/' + data[FILE_COLUMN]
  image = tf.image.decode_jpeg(tf.io.read_file(image_dir), channels=3)
  image = tf.cast(image, tf.float32)
  image = tf.keras.applications.resnet50.preprocess_input(image)
  image = tf.image.resize_with_crop_or_pad(image, 178, 178)
  image = tf.image.resize(image, [target_dim, target_dim])
  # image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)

  return image, data[ID_COLUMN]


def load_data(mode, partition_file, identity_df):
  with open(partition_file, 'r') as f:
    partitions = json.load(f)

  id_mask = identity_df[PERSON_COLUMN].isin(partitions[mode])
  data = identity_df.loc[id_mask].copy()
  data[ID_COLUMN] = data.groupby(PERSON_COLUMN).ngroup()

  return data


def make_dataset(dir, data_dim, mode, batch_size, data_split):
  image_dir = os.path.join(dir, 'img_align_celeba')
  identity_file = os.path.join(dir, 'identity_CelebA.txt')
  identity_df = pd.read_csv(identity_file, sep=' ', header=None)

  # 1. Build partitions
  partition_dir = os.path.join(dir, 'celeba_partitions')
  partition_file = build_partitions(identity_df, partition_dir, data_split)
  print()

  datas = load_data(mode, partition_file, identity_df)

  if mode == 'train':
    train_dataset = tf.data.Dataset.from_tensor_slices(dict(datas))
    train_dataset = train_dataset.map(
        lambda data: preprocess(data, image_dir, data_dim),
        num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(1)
    train_dataset = train_dataset.shuffle(1000)

    return train_dataset
  elif mode == 'test':
    test_dataset = tf.data.Dataset.from_tensor_slices(dict(datas))
    test_dataset = test_dataset.map(
        lambda data: preprocess(data, data_dim),
        num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
    test_dataset = test_dataset.prefetch(1)

    return test_dataset
  else:
    raise ValueError(f'Data mode {mode} not recognized')


if __name__ == '__main__':
  # Test
  dataset = make_dataset('data\\celeb_a', 80, 'train', 32, (.7, .15, .15))
  print(dataset)
