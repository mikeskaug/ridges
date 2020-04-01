
from tensorflow.keras import backend as K
import tensorflow as tf


def dice_loss(y_true, y_pred):
    intersection = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    union = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

    return 1 - intersection  / union


def keras_lovasz_hinge(labels, logits):
    return lovasz_hinge(logits, labels, per_image=True, ignore=None)


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    '''
    from https://github.com/bermanmaxim/LovaszSoftmax/blob/master/tensorflow/lovasz_losses_tf.py
    See The Lovász-Softmax loss... https://arxiv.org/abs/1705.08790
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    '''
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    '''
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    '''

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name='descending_sort')
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name='loss_non_void')
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name='loss'
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    '''
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    '''
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels


def focal_loss(y_true, y_pred):
    '''
    See "Focal Loss for Dense Object Detection" https://arxiv.org/abs/1708.02002
    '''
    gamma=2
    alpha=0.9
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    eps = 1e-4
    pt_1 = K.clip(pt_1, eps, 1 - eps)
    pt_0 = K.clip(pt_0, eps, 1 - eps)

    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def balanced_cross_entropy(y_true, y_pred):
    '''
    Compute the balanced cross entropy. Good for cases of large class imbalance.

    To more heavily penalize false negatives, set alpha > 0.5.
    To decrease false positivees, set alpha < 0.5
    '''
    alpha = 0.9
    # setting false locations to ones_like or zeros_like will result in log(pt_1) or log(1-pt_0) going to zero
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    # I've seen this other places. Assume it's to avoid possibly log(0).
    eps = 1e-4
    pt_1 = tf.clip_by_value(pt_1, eps, 1 - eps)
    pt_0 = tf.clip_by_value(pt_0, eps, 1 - eps)

    # reduce_sum eliminates the size=1 channel axis so the result is [batch, width, height]
    return -tf.reduce_sum(alpha * K.log(pt_1), axis=-1) - tf.reduce_sum((1 - alpha) * K.log(1. - pt_0), axis=-1)


def bce_plus_dice(y_true, y_pred):
    return balanced_cross_entropy(y_true, y_pred) + tf.reshape(dice_loss(y_true, y_pred), (-1, 1, 1))