import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Flatten, Dropout, Dense
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


class PrimaryCaps(Layer):
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


class Capsule3D(Layer):
  def __init__(self, n_caps, n_caps_dims, k, s, **kwargs):
    super(Capsule3D, self).__init__(**kwargs)
    self.C = n_caps
    self.L = n_caps_dims
    self.k = k
    self.s = s

  def build(self, input_shape):
    inp_h, inp_w, inp_C, inp_L = input_shape[1:]

    self.w = self.add_weight(
        shape=(inp_L, self.k, self.k, 1, self.C*self.L),
        initializer='glorot_uniform',
        name='w'
    )

    out_h = tf.cast((inp_h - self.k) / self.s + 1, tf.int32)
    out_w = tf.cast((inp_w - self.k) / self.s + 1, tf.int32)

    self.b = self.add_weight(
        shape=(self.C*out_h*out_w, self.L),
        initializer='zeros',
        name='b'
    )

  def call(self, input):                                # B, inp_h, inp_w, inp_C, inp_L
    b, inp_h, inp_w, inp_C, inp_L = input.shape[0:]

    u = tf.transpose(input, [0, 3, 4, 1, 2])            # B, inp_C, inp_L, inp_h, inp_w
    u = tf.reshape(                                     # B, inp_C*inp_L, inp_h, inp_w
        u, [-1, inp_C*inp_L, inp_h, inp_w, 1])

    u_hat = tf.nn.conv3d(                               # B, inp_C, out_h, out_w, C*L
        input=u,
        filters=self.w,
        strides=(1, inp_L, self.s, self.s, 1),
        padding='VALID')

    out_h, out_w = u_hat.shape[2:4]

    u_hat = tf.reshape(                                 # B, inp_C, out_h*out_w*C, L
        u_hat, [b, inp_C, -1, self.L])

    a = tf.einsum(                                      # B, inp_C, out_h*out_w*C, 1
        '...ijk,...pjk->...ij', u_hat, u_hat)[..., None]
    a = a / tf.sqrt(tf.cast(self.L, tf.float32))
    a = tf.nn.softmax(a, axis=-2)

    s = tf.reduce_sum(                                  # B, out_h*out_w*C, L
        u_hat * (a + self.b), axis=1)
    v = tf.reshape(                                     # B, out_h, out_w, C, L
        s, [-1, out_h, out_w, self.C, self.L])

    v = squash(v)
    return v


class Capsule2D(Layer):
  def __init__(self, C, L, **kwargs):
    super(Capsule2D, self).__init__(**kwargs)
    self.C = C
    self.L = L

  def build(self, input_shape):
    inp_h, inp_w, inp_C, inp_L = input_shape[1:]

    self.w = self.add_weight(
        shape=(self.C, inp_h*inp_w*inp_C, inp_L, self.L),
        initializer='glorot_uniform',
        name='w')

    self.b = self.add_weight(
        shape=(self.C, inp_h*inp_w*inp_C, 1),
        initializer='zeros',
        name='b')

  def call(self, input):
    inp_h, inp_w, inp_C, inp_L = input.shape[1:]

    u = tf.reshape(input, shape=(
        -1, inp_h*inp_w*inp_C, inp_L))

    u_hat = tf.einsum(
        '...ij,kijl->...kil', u, self.w)

    a = tf.einsum(                          # Calculate attention
        '...ij,...kj->...i', u_hat, u_hat)[..., None]
    a = a / tf.sqrt(tf.cast(self.L, tf.float32))
    a = tf.nn.softmax(a, axis=1)

    v = tf.reduce_sum(u_hat*(a + self.b), axis=-2)
    v = squash(v)

    return v


class Extractor(Model):
  def __init__(self, params, **kwargs):
    super(Extractor, self).__init__(**kwargs)

    self.params = params

    self.caps1 = PrimaryCaps(8, 32, 9, 1)
    self.caps2 = Capsule3D(16, 16, 3, 1)
    self.caps3 = Capsule2D(32, 8)

    self.regressor = Sequential(
        layers=[
            Flatten(),
            Dropout(0.5),
            Dense(10177, activation='relu')
        ])

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

  def call(self, input):
    image_a, image_b = input

    feat_a = self.feat(image_a)
    feat_b = self.feat(image_b)

    caps1_a = self.caps1(feat_a)
    caps1_b = self.caps1(feat_b)

    caps2_a = self.caps2(caps1_a)
    caps2_b = self.caps2(caps1_b)

    caps3_a = self.caps3(caps2_a)
    caps3_b = self.caps3(caps2_b)

    caps3_a_length = self.regressor(caps3_a)
    caps3_b_length = self.regressor(caps3_b)

    return caps3_a_length, caps3_b_length, caps3_a, caps3_b
