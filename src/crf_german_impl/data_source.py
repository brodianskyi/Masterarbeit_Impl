import numpy as np
import tensorflow as tf
from keras import backend as K

tf.enable_eager_execution()

# consider the case if there is the same number of tags in all tag sequences
n_tags = 4
batch_size = 2
max_seq_len = 5
embedding_dim = 3
input_shape = (batch_size, max_seq_len, embedding_dim)

# shape of X = (batch_size, max_seq_len, embedding_dim) -> (2, 5, 3)
X = tf.constant(
    [
        [
            [7, 4, 3],
            [3, 8, 6],
            [5, 9, 9],
            [8, 11, 12],
            [10, 4, 10]
        ]
        ,
        [
            [4, 8, 10],
            [3, 9, 5],
            [2, 4, 7],
            [5, 10, 8],
            [3, 7, 11]
        ]
    ]
    , dtype="float32"
)

# shape of mask = (batch_size, max_seq_len) = (2, 5)
mask = tf.constant([[True, True, True, False, False], [True, True, True, True, False]], dtype="bool")

# shape of kernel = (embedding_dim, n_tags) = (3, 4)
kernel = K.cast(tf.constant(np.random.randint(1, 13, size=(3, 4))), K.floatx())

# There is two sequences.
# The first sequence called (otr).
# The second sequence called (emb).
# shape of chain_kernel_otr = (n_tags, n_tags) = (4, 4)
chain_kernel_otr = tf.constant(np.arange(1, 17, dtype=np.float32), shape=[4, 4])
# shape of chain_kernel_emb = (n_tags, n_tags) = (4, 4)
chain_kernel_emb = tf.constant(np.arange(3, 19, dtype=np.float32), shape=[4, 4])

# chain_kernel for connections between sequence otr and emb
# shape = (n_tags, n_tags) = (4, 4)
chain_kernel_otr_emb = tf.constant(np.arange(5, 21, dtype=np.float32), shape=[4, 4])

# shape of bias = (n_tags,) = (4,)
bias = tf.constant(np.arange(1, 5, dtype=np.float32), shape=[4, ])

# shape of left_boundary =  (n_tags,) = (4,)
left_boundary = tf.constant(np.arange(2, 6, dtype=np.float32), shape=[4, ])

# shape of left_boundary =  (n_tags,) = (4,)
right_boundary = tf.constant(np.arange(4, 8, dtype=np.float32), shape=[4, ])

# data for crf_loss
# shape of y_true_1 = (batch_size, max_seq_len, n_tags)=(2, 5, 4)
y_true = tf.constant(
    [
        [
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0]
        ]
        ,
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ]
    ]
    , dtype="float32"
)

# shape of y_true = y_pred = (batch_size, max_seq_len, n_tags)=(2, 5, 4)

y_true_2 = tf.constant(
    [
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0]
        ]
        ,
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ]
    ]
    , dtype="float32"
)


y_pred_1 = tf.constant(
    [
        [
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ]
        ,
        [
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ]
    ]
    , dtype="float32"
)

y_pred_2 = tf.constant(
    [
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0]
        ]
        ,
        [
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ]
    ]
    , dtype="float32"
)

'''
n_tags = 4
batch_size = 2
max_seq_len = 5
embedding_dim = 3
input_shape = (batch_size, max_seq_len, embedding_dim)

# X = (2,5,3)
X = tf.constant(
    [
        [
            [7, 4, 3],
            [3, 8, 6],
            [5, 9, 9],
            [8, 11, 12],
            [10, 4, 10]
        ]
        ,
        [
            [4, 8, 10],
            [3, 9, 5],
            [2, 4, 7],
            [5, 10, 8],
            [3, 7, 11]
        ]
    ]
    , dtype="float32"
)
# mask shape=(batch_size, max_seq_len) = (2,5)
mask = tf.constant([[True, True, True, False, False], [True, True, True, True, False]], dtype="bool")

# shape = (embedding_dim, n_tags) -> shape=(3, 4)
kernel = K.cast(tf.constant(np.random.randint(11, 550, size=(3, 4))), K.floatx())

# shape = (n_tags, n_tags) -> (4, 4)
chain_kernel = K.cast(tf.constant(np.random.randint(1, 1000, size=(4, 4))), K.floatx())

# shape= (n_tags,) -> (4,)
bias = K.cast(tf.constant(np.random.randint(1, 500, size=(4,))), K.floatx())

# shape= (n_tags,) -> (4,)
left_boundary = K.cast(tf.constant(np.random.randint(10, 300, size=(4,))), K.floatx())

# shape = (n_tags,) -> (4,)
right_boundary = K.cast(tf.constant(np.random.randint(10, 300, size=(4,))), K.floatx())
'''
# argmin_table shape = (2, 5, 4)


argmin_table = tf.constant(
    [
        [
            [0, 1, 2, 3],
            [3, 2, 1, 0],
            [0, 1, 2, 3],
            [3, 2, 1, 0],
            [0, 1, 2, 3]
        ]
        ,
        [
            [1, 3, 2, 0],
            [1, 3, 2, 0],
            [2, 3, 3, 1],
            [0, 3, 0, 0],
            [2, 2, 2, 1]
        ]
    ]
    , dtype="float32"
)

