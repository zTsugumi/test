import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras_vggface.vggface import VGGFace
from capsnet.utils import ops_tensor

# EPS = tf.keras.backend.epsilon()


class Branch(Layer):
  def __init__(self, feat_dim, **kwargs):
    super(Branch, self).__init__(**kwargs)
    self.fc1 = Dense(feat_dim, activation='relu')
    self.fc2 = Dense(feat_dim, activation='relu')

    self.rot = ops_tensor.MLP([3, 1], 'relu', 'glorot_uniform', False, True)

  def call(self, input):
    x, rot = input
    x = self.fc1(x)
    x = self.fc2(x)

    rot = self.rot(rot)

    residual = rot * x
    # residual = rot * x + EPS

    return residual


class Extractor(Model):
  def __init__(self, params, **kwargs):
    super(Extractor, self).__init__(**kwargs)

    self.params = params

  def build(self, input_shape):
    img_shape = input_shape

    vggface = VGGFace(
        model='senet50',
        include_top=False,
        input_tensor=None,
        input_shape=img_shape[1:])
    feat = Flatten()(vggface.get_layer('avg_pool').output)

    if self.params['type'] == 'end2end1':
      self.branch_l = Branch(feat.shape[-1])
      self.branch_b = Branch(feat.shape[-1])
      self.branch_f = Branch(feat.shape[-1])

    self.extractor = Model(
        inputs=vggface.input,
        outputs=feat
    )

    self.regressor = Sequential(
        layers=[
            Dense(2048),
            BatchNormalization(),
            Activation('tanh'),
            Dropout(rate=0.2),
            Dense(self.params['n_class'], activation='softmax')])

  def call(self, input):
    if self.params['type'] == 'end2end1':
      x, rot_mat = input
    else:
      x = input

    feat_tmp = self.extractor(x)

    if self.params['type'] == 'end2end1':
      rot_mat = tf.reshape(rot_mat, [-1, 3, 3])
      l, b, f = tf.unstack(rot_mat, axis=1)

      residual_l = self.branch_l((feat_tmp, l))
      residual_b = self.branch_l((feat_tmp, b))
      residual_f = self.branch_l((feat_tmp, f))

      feat = residual_l + residual_b + residual_f + feat_tmp

      # residual_l = self.branch_l((feat_tmp, l))
      # feat_tmp = residual_l + feat_tmp

      # residual_b = self.branch_b((feat_tmp, b))
      # feat_tmp = residual_b + feat_tmp

      # residual_f = self.branch_f((feat_tmp, f))
      # feat = residual_f + feat_tmp
    else:
      feat = feat_tmp

    feat = self.regressor(feat)

    return feat
