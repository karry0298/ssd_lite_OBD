import tensorflow as tf
from tensorflow.keras import layers,models,applications,preprocessing,optimizers

IMG_SHAPE = (None,None,3)
base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE,
        include_top = False,weights = 'imagenet')
base_model.trainable = False

class MobileNet(tf.keras.layers.Layer):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")
        self.layer1 = tf.keras.layers.Conv2D(filters=128*4,
                                            kernel_size=(1, 1),
                                            strides=2,
                                            padding="same")
        self.layer2 = tf.keras.layers.Conv2D(filters=256*4,
                                            kernel_size=(1, 1),
                                            strides=2,
                                            padding="same")
    def call(self, inputs, training=None, **kwargs):
        x= base_model(inputs)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        branch1 = self.layer1(x)
        branch2 = self.layer2(branch1)
        return branch1, branch2
