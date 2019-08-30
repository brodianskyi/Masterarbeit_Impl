import unittest
import tensorflow as tf
import src.crf_german_impl.data_source as d_src
from src.crf_german_impl.crf_impl import CRF
from src.crf_german_impl.factorial_crf_impl import FCRF
from src.crf_german_impl.crf_loss_impl import crf_loss
from src.crf_german_impl.crf_accuracies_impl import crf_viterbi_accuracy


class LayerTest(unittest.TestCase):
    tf.enable_eager_execution()

    def test_crf_layer(self):
        n_tags = d_src.n_tags

        crf = CRF(n_tags)
        # build(input_shape)
        crf.build(d_src.input_shape)
        # initialize weight for build()
        crf.kernel = d_src.kernel
        crf.chain_kernel = d_src.chain_kernel_otr
        crf.bias = d_src.bias
        # boundary: left_boundary
        crf.left_boundary = d_src.left_boundary
        # boundary: right_boundary
        crf.right_boundary = d_src.right_boundary
        # call(X, mask)
        crf.call(d_src.X, d_src.mask)
        crf_loss(d_src.y_true, d_src.y_pred_1, crf)
        crf.get_negative_log_likelihood(d_src.y_true, d_src.X, d_src.mask)
        # crf_viterbi_accuracy(d_src.y_true_1, d_src.y_pred_1, crf)

    def test_factorial_crf(self):
        fcrf = FCRF(d_src.n_tags)
        fcrf.build(d_src.input_shape)
        # initialize kernels
        fcrf.kernel = d_src.kernel
        fcrf.chain_kernel_otr = d_src.chain_kernel_otr
        fcrf.chain_kernel_emb = d_src.chain_kernel_emb
        fcrf.chain_kernel_otr_emb = d_src.chain_kernel_otr_emb
        # initialize bias
        fcrf.bias = d_src.bias
        # initialize boundary
        fcrf.left_boundary = d_src.left_boundary
        fcrf.right_boundary = d_src.right_boundary

        fcrf.call(d_src.X, d_src.mask)



