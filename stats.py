import tensorflow as tf


def levenshtein_distance(truth, hyp):
    zeros = tf.zeros((truth.shape[0], 1), dtype=tf.int64)
    ranges = tf.reshape(tf.range(truth.shape[0], dtype=tf.int64), (truth.shape[0], 1))
    indices = tf.concat([zeros, ranges], axis=1)
    hyp_st = tf.SparseTensor(indices, hyp, [1, 1])
    truth_st = tf.SparseTensor(indices, truth, [1, 1])
    return tf.edit_distance(hyp_st, truth_st, normalize=False)
