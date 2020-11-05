import tensorflow as tf
from core.models.resnet import MobileNet
from configuration import NUM_CLASSES, ASPECT_RATIOS
from tensorflow.keras.layers import Activation,BatchNormalization,Conv2D,DepthwiseConv2D

class SSD(tf.keras.Model):
    def __init__(self):
        super(SSD, self).__init__()
        self.num_classes = NUM_CLASSES
        self.anchor_ratios = ASPECT_RATIOS

        self.backbone = MobileNet()
        self.learnable_factor = self.add_weight(shape=(1, 1, 1, 512), dtype=tf.float32, initializer=tf.keras.initializers.Ones(), trainable=True)
        # self.conv1 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(1, 1), strides=1, padding="same")
        self.conv2_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1),  padding="same")
        self.conv2_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same")
        
        self.conv3_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1),  padding="same")        
        self.conv3_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same")
        
        self.conv4_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1),  padding="same")
        self.conv4_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),  padding="same")
        self.pool = tf.keras.layers.GlobalAveragePooling2D()

        self.predict_1 = self._predict_layer(k=self._get_k(i=0))
        self.predict_2 = self._predict_layer(k=self._get_k(i=1))
        self.predict_3 = self._predict_layer(k=self._get_k(i=2))
        self.predict_4 = self._predict_layer(k=self._get_k(i=3))
        self.predict_5 = self._predict_layer(k=self._get_k(i=4))
        self.predict_6 = self._predict_layer(k=self._get_k(i=5))

    def _predict_layer(self, k):
        filter_num = k * (self.num_classes + 4)
        return tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3,3), strides=1, padding="same")

    def _get_k(self, i):
        # k is the number of boxes generated at each position of the feature map.
        return len(self.anchor_ratios[i]) + 1

    def call(self, inputs, training=None, mask=None):
        branch_1, x = self.backbone(inputs, training=training)
        branch_1 = tf.math.l2_normalize(x=branch_1, axis=-1, epsilon=1e-12) * self.learnable_factor
        dwcv1 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, activation=None,use_bias=False, padding='same', name='1_dwconv2')(branch_1)
        b1 = BatchNormalization(momentum=0.99,name='1_sepconv2_bn')(dwcv1)
        a1 = Activation('relu', name='1_sepconv2_act')(b1)
        predict_1 = self.predict_1(a1)


        # x = self.conv1(x)
        branch_2 = x
        dwcv2 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, activation=None,use_bias=False, padding='same', name='2_dwconv2')(branch_2)
        b2 = BatchNormalization(momentum=0.99,name='2_sepconv2_bn')(dwcv2)
        a2 = Activation('relu', name='2_sepconv2_act')(b2)
        predict_2 = self.predict_2(a2)


        x = tf.nn.relu(self.conv2_1(x))
        bn_bef3 = BatchNormalization(momentum=0.99,name='bef_BN_3')(x)
        af_bef3 = Activation('relu', name='Bef_AF_3')(bn_bef3)
        x = tf.nn.relu(self.conv2_2(af_bef3))
        branch_3 = x
        dwcv3 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, activation=None,use_bias=False, padding='same', name='3_dwconv2')(branch_3)
        b3 = BatchNormalization(momentum=0.99,name='3_sepconv2_bn')(dwcv3)
        a3 = Activation('relu', name='3_sepconv2_act')(b3)
        predict_3 = self.predict_3(a3)


        x = tf.nn.relu(self.conv3_1(x))
        bn_bef4 = BatchNormalization(momentum=0.99,name='bef_BN_4')(x)
        af_bef4 = Activation('relu', name='Bef_AF_4')(bn_bef4)
        x = tf.nn.relu(self.conv2_2(af_bef4))
        branch_4 = x
        dwcv4 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, activation=None,use_bias=False, padding='same', name='4_dwconv2')(branch_4)
        b4 = BatchNormalization(momentum=0.99,name='4_sepconv2_bn')(dwcv4)
        a4 = Activation('relu', name='4_sepconv2_act')(b4)
        predict_4 = self.predict_4(a4)


        x = tf.nn.relu(self.conv4_1(x))
        bn_bef5 = BatchNormalization(momentum=0.99,name='bef_BN_3')(x)
        af_bef5 = Activation('relu', name='Bef_AF_3')(bn_bef5)
        x = tf.nn.relu(self.conv2_2(af_bef5))
        branch_5 = x
        dwcv5 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, activation=None,use_bias=False, padding='same', name='5_dwconv2')(branch_5)
        b5 = BatchNormalization(momentum=0.99,name='5_sepconv2_bn')(dwcv5)
        a5 = Activation('relu', name='5_sepconv2_act')(b5)
        predict_5 = self.predict_5(a5)


        branch_6 = self.pool(x)
        branch_6 = tf.expand_dims(input=branch_6, axis=1)
        branch_6 = tf.expand_dims(input=branch_6, axis=2)
        dwcv6 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, activation=None,use_bias=False, padding='same', name='6_dwconv2')(branch_6)
        b6 = BatchNormalization(momentum=0.99,name='6_sepconv2_bn')(dwcv6)
        a6 = Activation('relu', name='6_sepconv2_act')(b6)
        predict_6 = self.predict_6(a6)

        # predict_i shape : (batch_size, h, w, k * (c+4)), where c is self.num_classes.
        # h == w == [38, 19, 10, 5, 3, 1] for predict_i (i: 1~6)
        return [predict_1, predict_2, predict_3, predict_4, predict_5, predict_6]


def ssd_prediction(feature_maps, num_classes):
    batch_size = feature_maps[0].shape[0]
    predicted_features_list = []
    for feature in feature_maps:
        predicted_features_list.append(tf.reshape(tensor=feature, shape=(batch_size, -1, num_classes + 4)))
    predicted_features = tf.concat(values=predicted_features_list, axis=1)  # shape: (batch_size, 8732, (c+4))
    return predicted_features
