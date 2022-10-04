import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, AveragePooling2D, Conv2D, Dense, Reshape
from tensorflow.keras.applications.resnet50 import ResNet50
from capsnet.utils import ops_tensor

EPS = tf.keras.backend.epsilon()


# Tested
class ResNet(Layer):
  '''
  Output of the 1st, 2nd and 3rd block of Resnet50
  Input: (B, 64, 64, 3)
  Output: [(B, 8, 8, 64), (B, 8, 8, 64), (B, 8, 8, 64)]
  '''

  def __init__(self, **kwargs):
    super(ResNet, self).__init__(**kwargs)

  def build(self, input_shape):
    resnet = ResNet50(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape[1:],
        pooling=None,
        classes=1000)

    feat1 = resnet.get_layer('conv2_block3_out').output   # (None, 16, 16, 256)
    feat1 = Conv2D(64, (1, 1))(feat1)
    feat1 = AveragePooling2D((2, 2))(feat1)

    feat2 = resnet.get_layer('conv3_block2_out').output   # (None, 8, 8, 512)
    feat2 = Conv2D(64, (1, 1))(feat2)

    feat3 = resnet.get_layer('conv3_block4_out').output   # (None, 8, 8, 512)
    feat3 = Conv2D(64, (1, 1))(feat3)

    self.model = Model(
        inputs=resnet.input,
        outputs=[feat1, feat2, feat3])

  def call(self, input):
    return self.model(input)


# Tested
class PrimaryCapsule(Layer):
  '''
  Take an input as a list of 3 stages of a feature map and select a set
  of meaningful features through fine-grained structure feature aggregation.
  The output is a set of primary capsule.

  Input: [(B, 8, 8, 64), (B, 8, 8, 64), (B, 8, 8, 64)]
  Output: (B, n_caps, n_caps_dim)
  '''

  def __init__(self, n_caps, n_caps_dim, **kwargs):
    super(PrimaryCapsule, self).__init__(**kwargs)
    self.n_caps = n_caps
    self.n_caps_dim = n_caps_dim

  def build(self, input_shape):
    _, h, w, _ = input_shape[0][:]

    self.SR = Sequential(
        name='SR',
        layers=[
            Dense(self.n_caps_dim * h * w * 3, activation='sigmoid'),
            Reshape([self.n_caps_dim, h * w * 3])
        ]
    )

    self.SL = Sequential(
        name='SL',
        layers=[
            Dense(int(self.n_caps / 3) * self.n_caps_dim, activation='sigmoid'),
            Reshape([int(self.n_caps / 3), self.n_caps_dim])
        ]
    )

    self.built = True

  def get_attention_map(self, feature):
    b = feature.shape[0]

    # (B, 8, 8)
    _, a = tf.nn.moments(feature, axes=-1)  # Attention map

    # (B, 64)
    a = tf.reshape(a, [b, -1])

    # (B, caps_dim, 8*8*3)
    SR = self.SR(a)

    return a, SR

  def call(self, input):
    # s1: stage 1
    feat_s1, feat_s2, feat_s3 = input[0], input[1], input[2]

    b, h, w = feat_s1.shape[:-1]

    # 1. There are 3 options here to build attention map: Variance, Conv2D or ...
    # (B, 64), (B, caps_dim, 8*8*3)
    a_s1, SR_s1 = self.get_attention_map(feat_s1)
    a_s2, SR_s2 = self.get_attention_map(feat_s2)
    a_s3, SR_s3 = self.get_attention_map(feat_s3)

    # (B, 64*3)
    a_all_s = tf.concat([a_s1, a_s2, a_s3], axis=-1)

    # (B, n_caps/3, caps_dim)
    SL = self.SL(a_all_s)

    # (B, n_caps/3, 8*8*3)
    S_s1, S_s2, S_s3 = [tf.matmul(SL, SR) for SR in [SR_s1, SR_s2, SR_s3]]

    # (B, n_caps/3, 64)
    S_s1_norm, S_s2_norm, S_s3_norm = [
        tf.tile(tf.reduce_sum(S, axis=-1, keepdims=True) + EPS, [1, 1, 64])
        for S in [S_s1, S_s2, S_s3]]

    # 2. Now that we have S, multiply it with input feature
    #    to reduce its dimensionality
    # (B, 8*8, 64)
    feat_s1 = tf.reshape(feat_s1, [b, h*w, -1])
    feat_s2 = tf.reshape(feat_s2, [b, h*w, -1])
    feat_s3 = tf.reshape(feat_s3, [b, h*w, -1])

    # (B, 8*8*3, 64)
    feat_all_s = tf.concat([feat_s1, feat_s2, feat_s3], axis=1)

    # (B, n_caps/3, 64)
    caps_s1 = tf.matmul(S_s1, feat_all_s) / S_s1_norm
    caps_s2 = tf.matmul(S_s2, feat_all_s) / S_s2_norm
    caps_s3 = tf.matmul(S_s3, feat_all_s) / S_s3_norm

    # (B, n_caps, 64)
    primary_caps = tf.concat([caps_s1, caps_s2, caps_s3], axis=1)

    return primary_caps


# Tested
class ClassCapsule(Layer):
  '''
  Normal routing algorithm
  '''

  def __init__(self, n_caps, n_caps_dim, r=3, **kwargs):
    super(ClassCapsule, self).__init__(**kwargs)
    self.n_caps = n_caps
    self.n_caps_dim = n_caps_dim
    self.r = r

  def build(self, input_shape):
    in_n_caps, in_n_caps_dim = input_shape[1], input_shape[2]

    self.W = self.add_weight(
        name='W',
        shape=[in_n_caps, in_n_caps_dim, self.n_caps*self.n_caps_dim],
        initializer='glorot_uniform',
    )

    self.bias = self.add_weight(
        name='bias',
        shape=[self.n_caps, self.n_caps_dim],
        initializer='zeros',
    )

    self.built = True

  def squash(self, s):
    '''
    Squash activation
    '''
    norm = ops_tensor.safe_norm(s, axis=-1)
    norm_squared = tf.square(norm)
    return norm_squared / (1.0 + norm_squared) / norm * s

  def call(self, input):
    in_n_caps = input.shape[1]

    u_hat = tf.einsum(                                        # (B, input_n_caps, n_caps*n_caps_dim)
        '...ij,ijk->...ik', input, self.W)
    u_hat = tf.reshape(u_hat, shape=(
        -1, in_n_caps, self.n_caps, self.n_caps_dim))         # (B, input_n_caps, n_caps, n_caps_dim)

    # Routing
    b = tf.zeros(tf.shape(u_hat)[:-1])[..., None]             # (B, input_n_caps, n_caps, 1)
    for r in range(self.r):
      c = tf.nn.softmax(b, axis=2)                            # (B, input_n_caps, n_caps, 1)
      s = tf.reduce_sum(
          u_hat*c, axis=1, keepdims=True)                     # (B, 1, n_caps, n_caps_dim)
      s += self.bias
      v = self.squash(s)                                      # (B, 1, n_caps, n_caps_dim)
      if r < self.r - 1:
        agreement = tf.reduce_sum(
            u_hat*v, axis=-1, keepdims=True)
        b += agreement
    v = tf.squeeze(v, axis=1)                                 # (B, n_caps, n_caps_dim)

    return v


# Tested
class SplitCapsule(Layer):
  '''
  Here we split all capsules into 3 groups (each of which contains several 16D capsule),
  then we upscale (using Dense) each of the dimension of each group to the power of 3.
  For example:
    input: (B, 3, 16)
    -> split it to [(B, 16), (B, 16), (B, 16)]
    -> upscale each to [(B, 3, 3), (B, 3, 9), (B, 3, 27)]
    -> merge them along last axis
  '''

  def __init__(self, n_caps, stage_base, **kwargs):
    super(SplitCapsule, self).__init__(**kwargs)
    self.n_caps = n_caps

    n_units = [3 * stage_base[i] ** (i+1) for i in range(len(stage_base))]

    self.fc = [Dense(n_unit, activation='sigmoid') for n_unit in n_units]

  def process_capsule(self, stage_idx, capsule):
    '''
    stage_idx:    0/1/2     current stage index
    capsule:      (B, 16)   1 capsule
    '''
    b = capsule.shape[0]

    upscale = self.fc[stage_idx](capsule)
    upscale = tf.reshape(upscale, [b, 3, -1])

    return upscale

  def call(self, input):
    b = input.shape[0]

    # 1. Split
    caps_s1 = input[:, 0:self.n_caps//3, :]                 # input[:, 0:1, :]
    caps_s1 = tf.reshape(caps_s1, [b, -1])

    caps_s2 = input[:, self.n_caps//3:2*self.n_caps//3, :]  # input[:, 1:2, :]
    caps_s2 = tf.reshape(caps_s2, [b, -1])

    caps_s3 = input[:, 2*self.n_caps//3:self.n_caps, :]     # input[:, 2:3, :]
    caps_s3 = tf.reshape(caps_s3, [b, -1])

    # 2. Upscale
    # (B, 3, 3), (B, 3, 9), (B, 3, 27)
    caps_s1 = self.process_capsule(0, caps_s1)
    caps_s2 = self.process_capsule(1, caps_s2)
    caps_s3 = self.process_capsule(2, caps_s3)

    # (B, 3, 39)
    caps = tf.concat([caps_s1, caps_s2, caps_s3], axis=-1)

    return caps


# WIP
class SSR(Layer):
  def __init__(self, stage_base, lambda_d, **kwargs):
    super(SSR, self).__init__(**kwargs)
    self.stage_base = stage_base
    self.lambda_d = lambda_d
    self.trainable = False

  def call(self, input):
    x = input
    a = x[:, :, 0] * 0
    b = x[:, :, 0] * 0
    c = x[:, :, 0] * 0

    # 3, 9, 27
    s1, s2, s3 = [self.stage_base[i] ** (i+1) for i in range(3)]

    di, dj, dk = [s // 2 for s in [s1, s2, s3]]

    # s1 = 3
    # i = 0, 1, 2 ~> i-di = -1, 0, 1
    for i in range(0, s1):
      a = a + (i - di) * x[:, :, i]
    a = a / (s1//2)

    # s2 = 9
    # j - dj ~> [-4, 4]
    for j in range(0, s2):
      b = b + (j - dj) * x[:, :, j+3]
    b = b / (s2//2)

    # s3 = 27
    for k in range(0, s3):
      c = c + (k - dk) * x[:, :, k+12]
    c = c / (s3//2)

    pred = (a+b+c) / 3

    return pred


class Branch(Layer):
  def __init__(self, params, **kwargs):
    super(Branch, self).__init__(**kwargs)

    self.caps1 = PrimaryCapsule(
        n_caps=params['pose_n_caps1'],
        n_caps_dim=params['pose_n_caps1_dim'])
    self.caps2 = ClassCapsule(
        n_caps=params['pose_n_caps2'],
        n_caps_dim=params['pose_n_caps2_dim'],
        r=params['pose_routing'])
    self.split_caps = SplitCapsule(
        n_caps=params['pose_n_caps2'],
        stage_base=params['pose_stage_base'])
    self.ssr = SSR(
        stage_base=params['pose_stage_base'],
        lambda_d=params['pose_lambda_d'])

  def call(self, input):
    # 1. Generate primary capsules
    # (B, 21, 64)
    caps1 = self.caps1(input)

    # 2. Generate class capsules
    # (B, 3, 16)
    caps2 = self.caps2(caps1)

    # 3. Split class capsules into 3 groups
    # (B, 3, 39)
    caps2 = self.split_caps(caps2)

    # 4. Taking the prediction, dynamic index shifting, and dynamic scaling
    # for the final regression output. In this case, the output is one of
    # (yaw, pitch, roll).
    # (B, 3)
    pred_vec = self.ssr(caps2)

    return pred_vec


class PoseEstimator(Model):
  def __init__(self, params, **kwargs):
    '''
    Here we have 3 branches, each branch is used to
    estimate 1 component of the rotation matrix.

    params:
      pose_lambda_d:    1             lambda??
      pose_stage_base:  [3, 3, 3]     base in each stage
      pose_n_caps1:     21            # primary capsule
      pose_n_caps1_dim: 5             primary capsule dimensionality
      pose_n_caps2:     3             # class capsule
      pose_n_caps2_dim: 16            class capsule dimensionality
      pose_routing:     2             # capsule routing
    '''
    super(PoseEstimator, self).__init__(**kwargs)

    self.params = params

    self.resnet = ResNet()

    self.branch0 = Branch(params)
    self.branch1 = Branch(params)
    self.branch2 = Branch(params)

  def call(self, input):
    # [(B, 8, 8, 64), (B, 8, 8, 64), (B, 8, 8, 64)]
    feat_list = self.resnet(input)

    pred_vec0 = self.branch0(feat_list)
    pred_vec1 = self.branch1(feat_list)
    pred_vec2 = self.branch2(feat_list)

    pred_vecs = tf.concat([pred_vec0, pred_vec1, pred_vec2], axis=-1)

    return pred_vecs
