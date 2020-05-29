
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


def iou(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(tf.cast((y_true + y_pred) >= 1, tf.float64))

    return intersection / union


def dice_coefficient(y_true, y_pred):
    intersection = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    union = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

    return (intersection + 1) / (union + 1)


def accuracy(y_true, y_pred):
    return tf.math.reduce_sum((y_true == y_pred).astype(float)) / y_true.size