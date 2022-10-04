from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.applications.resnet50 import ResNet50


class Extractor(Model):
  def __init__(self, params, **kwargs):
    super(Extractor, self).__init__(**kwargs)

    self.params = params

  def build(self, input_shape):
    img_shape = input_shape

    base_resnet = ResNet50(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=img_shape[1:],
        pooling=None,
    )
    # x = GlobalAveragePooling2D()(base_resnet.output)
    x = Flatten()(base_resnet.output)
    x = Dense(1000, activation='relu')(x)
    x = Dense(500, activation='sigmoid')(x)

    self.extractor = Model(
        inputs=base_resnet.input,
        outputs=x
    )

  def call(self, input):
    feat_a = self.extractor(input)

    return feat_a
