import unittest
import tensorflow as tf
from keras import backend as K
from src.crf_german_impl.crf_impl import CRF

class LayerTest(unittest.TestCase):

    def test_crf_layer(self):
        n_labels = 12
        energy = tf.constant(
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]
                ]
                ,
                [
                    [13, 14, 15, 16],
                    [17, 18, 19, 20],
                    [21, 22, 23, 24],
                ]
            ]
            , dtype="int32"
        )
        energy = K.cast(energy, K.floatx())
        crf = CRF(n_labels)
        crf.add_boundary_energy(energy, energy, energy, energy)