import tensorflow as tf
import numpy as np

n_tags = 4
batch_size = 2
max_seq_len = 5
embedding_dim = 3
# output from Embedding layer, input for CRF-Layer
input_shape = (batch_size, max_seq_len, embedding_dim)


# shape = (batch_size, max_seq_len, embedding_dim) -> (2, 5, 3)
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

# shape = (batch_size, max_seq_len) -> (2, 5)
mask = tf.constant([[True, True, False, True, False], [False, True, True, False, True]], dtype="bool")

# shape = (embedding_dim, n_tags) -> shape=(3, 4)
kernel = tf.constant(np.arange(1, 13, dtype=np.float32), shape=[3, 4])

# shape = (n_tags, n_tags) -> (4, 4)
chain_kernel = tf.constant(np.arange(1, 17, dtype=np.float32), shape=[4, 4])

# shape= (n_tags,) -> (4,)
bias = tf.constant(np.arange(1, 5, dtype=np.float32), shape=[4, ])

# shape= (n_tags,) -> (4,)
left_boundary = tf.constant(np.arange(1, 5, dtype=np.float32), shape=[4, ])

# shape = (n_tags,) -> (4,)
right_boundary = tf.constant(np.arange(1, 5, dtype=np.float32), shape=[4, ])

