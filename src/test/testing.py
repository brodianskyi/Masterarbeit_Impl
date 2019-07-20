import tensorflow as tf
from keras import backend as K

tf_initial_tensor_constant = tf.constant(
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

x = tf.compat.v1.placeholder(shape=(None, 4, 4), dtype='float32')
y = Flatten()(x)

