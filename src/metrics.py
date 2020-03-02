
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


def iou(y_true, y_pred):
    y_pred_ = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred_)
    union = tf.reduce_sum(tf.cast((y_true + y_pred_) >= 1, tf.float32))

    return intersection / union


def dice_coefficient(y_true, y_pred):
    intersection = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    union = tf.reduce_sum(y_true + y_pred, axis=-1)

    return (intersection + 1) / (union + 1)