import src.crf_german_impl.data_source as d_src
import tensorflow as tf

tf.enable_eager_execution()

def crf_nll(y_true, y_pred, crf):
    # implementation of the negative log-likelihood for CRF
    # used only in join mode
    # sparse_target is False by default
    X = d_src.X
    mask = d_src.mask
    nloglik = crf.get_negative_log_likelihood(y_true, X, mask)
    print("------------nloglik", nloglik)
    return nloglik


def crf_loss(y_true, y_pred, crf):
    # example for the join mode
    return crf_nll(y_true, y_pred, crf)
