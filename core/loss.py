import tensorflow as tf

from configuration import reg_loss_weight, NUM_CLASSES, alpha, gamma
from utils.focal_loss import sigmoid_focal_loss


# class SmoothL1Loss:
#     def __init__(self):
#         pass

#     def __call__(self, y_true, y_pred, *args, **kwargs):
#         absolute_value = tf.math.abs(y_true - y_pred)
#         mask_boolean = tf.math.greater_equal(x=absolute_value, y=1.0)
#         mask_float32 = tf.cast(x=mask_boolean, dtype=tf.float32)
#         smooth_l1_loss = (1.0 - mask_float32) * 0.5 * tf.math.square(absolute_value) + mask_float32 * (absolute_value - 0.5)
#         return smooth_l1_loss


class SSDLoss:
    def __init__(self):
        # self.smooth_l1_loss = SmoothL1Loss()
        self.reg_loss_weight = reg_loss_weight
        self.cls_loss_weight = 1 - reg_loss_weight
        self.num_classes = NUM_CLASSES

    # @staticmethod
    # def __cover_background_boxes(true_boxes):
    #     symbol = true_boxes[..., -1]
    #     mask_symbol = tf.where(symbol < 0.5, 0.0, 1.0)
    #     mask_symbol = tf.expand_dims(input=mask_symbol, axis=-1)
    #     cover_boxes_tensor = tf.tile(input=mask_symbol, multiples=tf.constant([1, 1, 4], dtype=tf.dtypes.int32))
    #     return cover_boxes_tensor

    def __call__(self, y_true, y_pred, *args, **kwargs):
        # y_true : tensor, shape: (batch_size, total_num_of_default_boxes, 5)
        # y_pred : tensor, shape: (batch_size, total_num_of_default_boxes, NUM_CLASSES + 4)
        true_class = tf.cast(x=y_true[..., -1], dtype=tf.dtypes.int32)
        pred_class = y_pred[..., :self.num_classes]
        true_class_onehot = tf.one_hot(indices=true_class, depth=self.num_classes, axis=-1)
        class_loss_value = tf.math.reduce_sum(sigmoid_focal_loss(y_true=true_class_onehot, y_pred=pred_class, alpha=alpha, gamma=gamma))

        symbol = y_true[..., -1]
        mask_symbol = tf.where(symbol < 0.5, 0.0, 1.0)
        mask_symbol = tf.expand_dims(input=mask_symbol, axis=-1)
        cover_boxes_tensor = tf.tile(input=mask_symbol, multiples=tf.constant([1, 1, 4], dtype=tf.dtypes.int32))

        cover_boxes = cover_boxes_tensor
        # true_coord = y_true[..., :4] * cover_boxes
        # pred_coord = y_pred[..., self.num_classes:] * cover_boxes
        true_coord = y_true[..., :4]
        pred_coord = y_pred[..., self.num_classes:]


        absolute_value = tf.math.abs(true_coord - pred_coord)
        mask_boolean = tf.math.greater_equal(x=absolute_value, y=1.0)
        mask_float32 = tf.cast(x=mask_boolean, dtype=tf.float32)
        smooth_l1_loss = (1.0 - mask_float32) * 0.5 * tf.math.square(absolute_value) + mask_float32 * (absolute_value - 0.5)
        # return smooth_l1_loss


        
        # reg_loss_value = tf.math.reduce_sum(self.smooth_l1_loss(y_true=true_coord, y_pred=pred_coord) * cover_boxes)
        reg_loss_value = tf.math.reduce_sum( smooth_l1_loss * cover_boxes )

        loss = self.cls_loss_weight * class_loss_value + self.reg_loss_weight * reg_loss_value
        return loss, class_loss_value, reg_loss_value
