import tensorflow as tf
import tensorflow.keras as keras

EPS = tf.keras.backend.epsilon()


def mse(y_true, y_pred):
  diff_sqr = tf.square(y_true - y_pred)
  return tf.reduce_sum(diff_sqr, axis=-1, keepdims=True)


def orth(_, y_pred):
  # v1_preds: (None, 3)
  v1_preds = y_pred[:, : 3]
  v2_preds = y_pred[:, 3: 6]
  v3_preds = y_pred[:, 6: 9]

  v1_preds = v1_preds / tf.norm(v1_preds, axis=-1, keepdims=True)
  v2_preds = v2_preds / tf.norm(v2_preds, axis=-1, keepdims=True)
  v3_preds = v3_preds / tf.norm(v3_preds, axis=-1, keepdims=True)

  v12_cross = tf.reduce_sum(tf.abs(v1_preds * v2_preds), axis=-1, keepdims=True)
  v13_cross = tf.reduce_sum(tf.abs(v1_preds * v3_preds), axis=-1, keepdims=True)
  v23_cross = tf.reduce_sum(tf.abs(v2_preds * v3_preds), axis=-1, keepdims=True)

  return v12_cross**2 + v13_cross**2 + v23_cross**2


def cross_entropy(y_true, y_pred):
  # label_a, label_b, _ = tf.unstack(y_true, axis=-1)
  # pred_a, pred_b = y_pred

  # label_a = tf.squeeze(label_a)
  # label_a = tf.cast(label_a, dtype=tf.int64)
  # label_a = tf.one_hot(label_a, 1000)

  # label_b = tf.squeeze(label_b)
  # label_b = tf.cast(label_b, dtype=tf.int64)
  # label_b = tf.one_hot(label_b, 1000)

  # loss_a = keras.losses.CategoricalCrossentropy()(label_a, pred_a + EPS)
  # loss_b = keras.losses.CategoricalCrossentropy()(label_b, pred_b + EPS)
  # return loss_a + loss_b

  label = tf.squeeze(y_true)
  label = tf.one_hot(label, 500)

  # loss = tf.keras.losses.MeanAbsoluteError()(label, y_pred)
  # loss = tf.reduce_sum(tf.abs(label - y_pred), axis=-1)
  loss = keras.losses.CategoricalCrossentropy()(label, y_pred + EPS)
  # loss  = keras.losses.SparseCategoricalCrossentropy()(label, y_pred + EPS)

  return loss


def cor_feat(x, y):
  mean_x = tf.reduce_mean(x, axis=-1, keepdims=True)
  mean_y = tf.reduce_mean(y, axis=-1, keepdims=True)
  dif_x = x - mean_x
  dif_y = y - mean_y

  numerator = tf.reduce_sum(dif_x * dif_y, axis=-1, keepdims=True)
  denominator = tf.sqrt(tf.reduce_sum(dif_x**2 * dif_y**2, axis=-1, keepdims=True) + EPS)

  return numerator / denominator


def contrastive(y_true, y_pred):
  _, _, label = tf.unstack(y_true, axis=-1)
  _, _, feat_a, feat_b = y_pred  # B, 10, 16

  # COR
  cor = cor_feat(feat_a, feat_b)
  cor = tf.squeeze(cor, axis=-1)

  label = tf.expand_dims(label, axis=-1)
  loss = (1 - label) * cor**2 + label * tf.maximum(1.7 - cor, 0)**2

  # sum_squared = tf.reduce_sum(tf.square(feat_a - feat_b), axis=1, keepdims=True)
  # sum_squared = tf.maximum(sum_squared, EPS)
  # euclidean_distance = tf.sqrt(tf.maximum(sum_squared, EPS))
  # squared_margin = tf.square(tf.maximum(1 - euclidean_distance, 0))

  # loss = tf.reduce_mean(y_true * sum_squared + (1 - y_true) * squared_margin)

  return loss


def sparse_cross_entropy(y_true, y_pred):
  print(y_true.shape)
  print(y_pred.shape)
  return keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred)


def margin(y_true, y_pred):
  y_true = tf.squeeze(y_true)
  y_true = tf.one_hot(y_true, 100)

  lamda = 0.5
  m_plus = 0.9
  m_minus = 0.1

  margin_left = tf.square(tf.maximum(0.0, m_plus - y_pred))
  margin_right = tf.square(tf.maximum(0.0, y_pred - m_minus))

  margin_left = y_true * margin_left
  margin_right = lamda * (1.0 - y_true) * margin_right

  L = margin_left + margin_right
  L = tf.reduce_mean(tf.reduce_sum(L, axis=-1))

  return L


def get_loss_funcs(loss_metrics):
  funcs = {}
  for key, loss_metric in loss_metrics.items():
    if loss_metric[0] == 'mse':
      funcs[key] = mse
    elif loss_metric[0] == 'orth':
      funcs[key] = orth
    elif loss_metric[0] == 'cross_entropy':
      funcs[key] = cross_entropy
    elif loss_metric[0] == 'contrastive':
      funcs[key] = contrastive
    elif loss_metric[0] == 'margin':
      funcs[key] = margin
    else:
      raise ValueError(f'Loss function {loss_metric[0]} not found')
  return funcs


def compute_loss(y_true, y_pred, loss_metrics):
  loss_funcs = get_loss_funcs(loss_metrics)
  losses = {}
  losses['total'] = 0.0
  for key, loss_func in loss_funcs.items():
    if key in ['pose', 'pose_orth', 'fe_verif', 'fe_iden']:
      losses[key] = loss_func(y_true, y_pred)

  for key, loss_metric in loss_metrics.items():
    losses['total'] += (losses[key] * loss_metric[1])

  # losses['total'] = sum(losses.values())

  return losses
