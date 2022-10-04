import os
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


def preprocess(image, label, target_dim):
  image = tf.expand_dims(image, -1)
  image = tf.repeat(image, 3, -1)
  image = tf.image.resize(image, [target_dim, target_dim])
  image = tf.cast(image, dtype=tf.float32) / 255.0
  label = tf.one_hot(label, depth=10)
  return image, label


def make_dataset(dir, data_dim, mode, batch_size):
  data_file = os.path.join(os.getcwd(), dir, 'mnist.npz')
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(data_file)

  if mode == 'train':
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.map(
        lambda image, label: preprocess(image, label, data_dim),
        num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(1)
    train_dataset = train_dataset.shuffle(1000)

    return train_dataset
  elif mode == 'test':
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.map(
        lambda image, label: preprocess(image, label, data_dim),
        num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
    test_dataset = test_dataset.prefetch(1)

    return test_dataset
  else:
    raise ValueError(f'Data mode {mode} not recognized')


if __name__ == '__main__':
  # Test
  dataset = make_dataset('data\\mnist', 'test', 32)
  print(dataset)
