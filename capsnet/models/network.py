import tensorflow as tf
from capsnet.utils import loss


class Network(object):
  def __init__(
      self,
      model,
      checkpoint_dir,
      optimizer_type,
      learning_rate
  ):
    # Setup optimizer
    self._learning_rate = learning_rate
    self._optimizer_type = optimizer_type
    self._create_optimizer()

    # Setup metric
    self._metric = tf.keras.metrics.CategoricalAccuracy()

    # Setup model
    self._model = model

    # Setup checkpoint
    self._create_checkpoint()
    self._update_checkpoint_dir(checkpoint_dir)

  @property
  def model(self):
    return self._model

  def _create_optimizer(self):
    if self._optimizer_type == 'adam':
      self._optimizer = tf.keras.optimizers.Adam(self._learning_rate, name='Optimizer')
    elif self._optimizer_type == 'sgd':
      self._optimizer = tf.keras.optimizers.SGD(self._learning_rate, name='Optimizer')
    elif self._optimizer_type == 'adamax':
      self._optimizer = tf.keras.optimizers.Adamax(self._learning_rate, name='Optimizer')
    else:
      raise ValueError(f'Optimizer {self._optimizer_type} not found')

  def _create_checkpoint(self):
    self._checkpoint = tf.train.Checkpoint(
        optimizer=self._optimizer,
        model=self._model,
        optimizer_step=tf.compat.v1.train.get_or_create_global_step()
    )

  def _update_checkpoint_dir(self, checkpoint_dir):
    self._manager = tf.train.CheckpointManager(
        self._checkpoint,
        directory=checkpoint_dir,
        max_to_keep=100)

  def restore(
      self,
      reset_optimizer=False,
      reset_global_step=False,
      epoch=None,
  ):
    if epoch is not None:
      status = self._checkpoint.restore(self._manager.directory + f'\ckpt-{epoch}')
    else:
      status = self._checkpoint.restore(self._manager.latest_checkpoint)

    try:
      status.assert_existing_objects_matched()
    except AssertionError as e:
      print('Error reload models: ', e)

    if reset_optimizer:
      self._create_optimizer()
      self._create_checkpoint()

    if reset_global_step:
      tf.compat.v1.train.get_or_create_global_step().assign(0)

  def save(self):
    self._manager.save()

  def train(
      self,
      batch,
      train_eager=True,
      loss_metrics=None,
  ):
    # Train step
    if train_eager:
      losses = self._train_step_eager(batch, loss_metrics)
    else:
      losses = self._train_step(batch, loss_metrics)

    log_update = losses
    if callable(self._learning_rate):
      log_update['learning_rate'] = self._learning_rate()
    else:
      log_update['learning_rate'] = self._learning_rate

    return log_update

  def validate(self, val_it):
    pass

  def test(self, test_it):
    for batch in test_it:
      x, y = batch[0], batch[1]
      score = self.model.evaluate(x, y)
      print(score)

  @tf.function
  def _train_step(
      self,
      batch,
      loss_metrics,
  ):
    return self._train_step_eager(batch, loss_metrics)

  def _train_step_eager(
      self,
      batch,
      loss_metrics,
  ):
    losses, variables, grads = self._loss_and_grad(
        batch,
        loss_metrics,
    )

    self._optimizer.apply_gradients(
        list(zip(grads, variables))
    )

    global_step = tf.compat.v1.train.get_or_create_global_step()
    tf.compat.v1.assign(global_step, global_step+1)

    return losses

  def _loss_and_grad(
      self,
      batch,
      loss_metrics,
  ):
    with tf.GradientTape() as tape:
      losses = self._compute_loss(batch, loss_metrics)
    variables = (self._model.trainable_variables)
    grads = tape.gradient(losses['loss_total'], variables)

    return losses, variables, grads

  def _compute_loss(
      self,
      batch,
      loss_metrics,
  ):
    x_true, y_true = batch[0], batch[1]
    y_pred = self._model(x_true)

    losses = loss.compute_loss(
        y_true, y_pred, loss_metrics)

    losses = {'loss_' + key: losses[key] for key in losses}

    # self._metric.update_state(y_true, y_pred)
    # losses['metric'] = self._metric.result()

    return losses
