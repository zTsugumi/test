import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Reshape, Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.applications.resnet import ResNet50

EPS = tf.keras.backend.epsilon()


def safe_norm(s, axis=-1, keepdims=True):
  '''
  Calculation of norm as tf.norm(), but here we add a small value of eps
  to the result to avoid 0
  '''
  s_ = tf.reduce_sum(tf.square(s), axis=axis, keepdims=keepdims)
  return tf.sqrt(s_ + EPS)


def squash(s):
  '''
  Squash activation
  '''
  norm = safe_norm(s, axis=-1)
  return (1.0 - 1.0/tf.exp(norm)) * (s / norm)


class PrimaryCaps(tf.keras.layers.Layer):
  '''
  This constructs a primary capsule layer using regular convolution layer
  '''

  def __init__(self, C, L, k, s, **kwargs):
    super(PrimaryCaps, self).__init__(**kwargs)
    self.C = C      # C: number of primary capsules
    self.L = L      # L: primary capsules dimension (num of properties)
    self.k = k      # k: kernel dimension
    self.s = s      # s: stride

  def get_config(self):
    config = {
        'C': self.C,
        'L': self.L,
        'k': self.k,
        's': self.s,
    }
    base_config = super(PrimaryCaps, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def compute_output_shape(self, input_shape):
    H, W = input_shape.shape[1:3]
    return (None, (H - self.k)/self.s + 1, (W - self.k)/self.s + 1, self.C, self.L)

  def build(self, input_shape):
    self.DW_Conv = tf.keras.layers.Conv2D(
        filters=self.C*self.L,
        kernel_size=self.k,
        strides=self.s,
        kernel_initializer='glorot_uniform',
        padding='valid',
        groups=self.C*self.L,
        activation='relu',
        name='conv'
    )
    self.built = True

  def call(self, input):
    x = self.DW_Conv(input)
    H, W = x.shape[1:3]
    x = tf.keras.layers.Reshape((H, W, self.C, self.L))(x)
    x = squash(x)
    return x


class DigitCaps(tf.keras.layers.Layer):
  '''
  This contructs the modified digit capsule layer
  '''

  def __init__(self, C, L, **kwargs):
    super(DigitCaps, self).__init__(**kwargs)
    self.C = C      # C: number of digit capsules
    self.L = L      # L: digit capsules dimension (num of properties)

  def get_config(self):
    config = {
        'C': self.C,
        'L': self.L
    }
    base_config = super(DigitCaps, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def compute_output_shape(self, input_shape):
    return (None, self.C, self.L)

  def build(self, input_shape):
    H = input_shape[1]
    W = input_shape[2]
    input_C = input_shape[3]
    input_L = input_shape[4]

    self.W = self.add_weight(               # Transformation matrix
        shape=(self.C, H*W*input_C, input_L, self.L),
        initializer='glorot_uniform',
        name='W'
    )
    self.bias = self.add_weight(               # Coupling Coefficient
        shape=(self.C, H*W*input_C, 1),
        initializer='zeros',
        name='bias'
    )

  def call(self, input):
    H, W, input_C, input_L = input.shape[1:]
    u = tf.reshape(input, shape=(
        -1, H*W*input_C, input_L))

    u_hat = tf.einsum(
        '...ij,kijl->...kil', u, self.W)

    a = tf.einsum(                          # Calculate attention
        '...ij,...kj->...i', u_hat, u_hat)[..., None]
    a = a / tf.sqrt(tf.cast(self.L, tf.float32))
    a = tf.nn.softmax(a, axis=1)

    s = tf.reduce_sum(u_hat*(a + self.bias), axis=-2)
    v = squash(s)

    return v


class Extractor(Model):
  def __init__(self, params, **kwargs):
    super(Extractor, self).__init__(**kwargs)

    self.params = params

    self.caps1 = PrimaryCaps(
        C=params['fe_n_caps1'],
        L=params['fe_n_caps1_dim'],
        k=9,
        s=1)

    self.caps2 = DigitCaps(
        C=params['fe_n_caps2'],
        L=params['fe_n_caps2_dim'])

  def build(self, input_shape):
    resnet = ResNet50(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape[0][1:],
        pooling=None,
        classes=1000
    )

    self.feat = Model(
        inputs=resnet.input,
        outputs=resnet.get_layer('conv2_block3_out').output
    )

    self.regressor = Sequential(
      layers=[
        Flatten(),
        Dropout(0.5),
        Dense(10177, activation='relu')
      ]
    )

  def call(self, input):
    image_a, image_b = input

    feat_a = self.feat(image_a)
    feat_b = self.feat(image_b)

    caps1_a = self.caps1(feat_a)
    caps1_b = self.caps1(feat_b)

    caps2_a = self.caps2(caps1_a)
    caps2_b = self.caps2(caps1_b)

    caps2_a_iden = self.regressor(caps2_a)
    caps2_b_iden = self.regressor(caps2_b)

    return caps2_a_iden, caps2_b_iden, caps2_a, caps2_b
