import unittest
import numpy as np
from src.crf_german_impl.crf_impl import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
import src.crf_german_impl.data_source as d_src


class LayerTest(unittest.TestCase):

    def test_crf_layer(self):
        n_tags = 12

        crf = CRF(n_tags + 1)
        # build(input_shape)
        crf.build(d_src.input_shape)
        # call(X, mask)
        # shape of X = input_shape
        crf.call(d_src.X, d_src.mask)
