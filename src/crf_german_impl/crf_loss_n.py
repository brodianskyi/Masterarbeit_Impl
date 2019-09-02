from keras import backend as K
from keras.losses import categorical_crossentropy
from keras.losses import sparse_categorical_crossentropy


def crf_nll(y_true, y_pred):
    """The negative log-likelihood for linear chain Conditional Random Field (CRF).

    This loss function is only used when the CRF-layer
    is trained in the "join" mode.

    """

    crf, idx = y_pred._keras_history[:2]
    if crf._outbound_nodes:
        raise TypeError('When learn_model="join", CRF must be the last layer.')
    if crf.sparse_target:
        y_true = K.one_hot(K.cast(y_true[:, :, 0], 'int32'), crf.units)
    X = crf._inbound_nodes[idx].input_tensors
    mask = crf._inbound_nodes[idx].input_masks
    nloglik = crf.get_negative_log_likelihood(y_true, X, mask)
    return nloglik


def crf_loss(y_true, y_pred):

    crf, idx = y_pred._keras_history[:2]
    if crf.learn_mode == 'join':
        return crf_nll(y_true, y_pred)
    else:
        if crf.sparse_target:
            return sparse_categorical_crossentropy(y_true, y_pred)
        else:
            return categorical_crossentropy(y_true, y_pred)
