import unittest
import numpy as np
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy


class LayerTest(unittest.TestCase):
    def test_crf_layer(self):
        # Hyperparameter setting
        vocab_size = 20
        n_classes = 11
        batch_size = 2
        maxlen = 2

        # Random features
        x = np.random.randint(1, vocab_size, size=(batch_size, maxlen))
        # Random tag indices
        y = np.random.randint(n_classes, size=(batch_size, maxlen))
