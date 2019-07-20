import unittest

import numpy as np
from keras.layers import Embedding, Input, Dense
from .crf_ger import CRFLayer

class LayerTest(unittest.TestCase):

    def test_crf_layer(self):
        vocab_size = 20
        n_classes = 11
        batch_size = 2
        maxlen = 2

        # Random features
        x = np.random.randint(1, vocab_size, size=(batch_size, maxlen))
        print("----x", x)

        # Random tag indices representing the gold sequence
        y = np.random.randint(n_classes, size=(batch_size, maxlen))
        print("---1y", y)
        # one hot encoded
        y = np.eye(n_classes)[y]
        print("---2y", y)

        # define the sequence
        # [2] * 2 = [2 2]
        # All sequences in this example have the same length, but they can be variable in a real model.
        s = np.array([maxlen] * batch_size, dtype="int32")
        print("---s", s)

        # Build a model
        # Input -  Instantiate a Keras tensor.
        # batch_shape = (2, 2)
        # batch_shape - the expected input will be batches of (2, 2) dimensional vectors
        word_ids = Input(batch_shape=(batch_size, maxlen), dtype="int32")
        # Embedding convert positive integers to the dense vectors [[4],[20]] -> [[0.25, 0.1], [0.6, -0.2]]
        # (vocab_size, output_dim); output_dim -  dimension of the dense embedding = number_of_classes
        word_embeddings = Embedding(vocab_size, n_classes)(word_ids)
        # batch_shape=[2, 1]
        sequence_length = Input(batch_shape=[batch_size, 1], dtype="int32")
        crf = CRFLayer()
        # will be called the method build
        pred = crf([word_embeddings, sequence_length])




