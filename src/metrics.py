
import tensorflow as tf
from keras import backend as K
import numpy as np


def mean_iou(y_true, y_pred):
    prec = []
    weights = tf.where(tf.equal(y_true, 0), x=0*tf.ones(tf.shape(y_true)), y=1.0*tf.ones(tf.shape(y_true)))
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2, weights=weights)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)