import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

EPS = keras.backend.epsilon()


def safe_norm(s, axis=-1, keepdims=True):
  '''
  Calculation of norm as tf.norm(), but here we add a small value of eps
  to the result to avoid 0
  '''
  s_ = tf.reduce_sum(tf.square(s), axis=axis, keepdims=keepdims)
  return tf.sqrt(s_ + EPS)


class MLP(keras.layers.Layer):
  def __init__(self, outs, activation, kernel_init, skip_connect, activate_last, **kwargs):
    super(MLP, self).__init__(**kwargs)

    self.layers = [
        Dense(out,
              activation=activation,
              kernel_initializer=kernel_init)
        for out in outs[:-1]]

    if activate_last:
      self.layers.append(Dense(outs[-1],
                               activation=activation,
                               kernel_initializer=kernel_init))
    else:
      self.layers.append(Dense(outs[-1],
                               activation=None,
                               kernel_initializer=kernel_init))

    self.skip_connect = skip_connect

  def get_config(self):
    config = {
        'layers': self.layers,
        'skip_connect': self.skip_connect,
    }
    base_config = super(MLP, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, input):
    out = self.layers.layers[0](input)

    for layer in self.layers.layers[1:]:
      if self.skip_connect:
        out = tf.concat([out, input], axis=-1)

      out = layer(out)

    return out
