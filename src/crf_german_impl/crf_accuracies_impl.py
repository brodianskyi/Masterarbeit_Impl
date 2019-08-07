import src.crf_german_impl.data_source as d_src
import tensorflow as tf
import keras.backend as K

tf.enable_eager_execution()


def _get_accuracy(y_true, y_pred, mask, sparse_target=False):
    y_pred = K.argmax(y_pred, -1)
    if sparse_target:
        y_true = K.cast(y_true[:, :, 0], K.dtype(y_pred))
    else:
        y_true = K.argmax(y_true, -1)
    judge = K.cast(K.equal(y_pred, y_true), K.floatx())
    if mask is None:
        return K.mean(judge)
    else:
        mask = K.cast(mask, K.floatx())
        return K.sum(judge * mask) / K.sum(mask)


def crf_viterbi_accuracy(y_true, y_pred, crf):
    X = d_src.X
    mask = d_src.mask
    y_pred = crf.viterbi_decoding(X, mask)
    print("----------_get_accuracy", _get_accuracy(y_true, y_pred, mask, crf.sparse_target))
    return _get_accuracy(y_true, y_pred, mask, crf.sparse_target)
