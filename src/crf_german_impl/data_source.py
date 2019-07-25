import tensorflow as tf
import numpy as np
from keras import backend as K

batch_size = 2
max_seq_len = 5
embedding_dim = 3
# output from Embedding layer, input for CRF-Layer
input_shape = (batch_size, max_seq_len, embedding_dim)

# shape = (2, 5, 3)
X = tf.constant(
    [
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15]
        ]
        ,
        [
            [16, 17, 18],
            [19, 20, 21],
            [22, 23, 24],
            [25, 26, 27],
            [28, 29, 30]
        ]
    ]
    , dtype="float32"
)

# shape = (2, 5)
mask = tf.constant([[True, True, False, True, False], [False, True, True, False, True]], dtype="bool")
