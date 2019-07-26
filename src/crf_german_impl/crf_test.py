import unittest
import numpy as np
from src.crf_german_impl.crf_impl import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
import src.crf_german_impl.data_source as d_src


class LayerTest(unittest.TestCase):

    def test_crf_layer(self):
        n_tags = d_src.n_tags

        crf = CRF(n_tags)
        # build(input_shape)
        crf.build(d_src.input_shape)

        # initialize weight for build()
        crf.kernel = d_src.kernel
        crf.bias = d_src.bias

        # boundary: left_boundary
        crf.left_boundary = d_src.left_boundary

        # boundary: right_boundary
        crf.right_boundary = d_src.right_boundary

        # call(X, mask)
        crf.call(d_src.X, d_src.mask)

