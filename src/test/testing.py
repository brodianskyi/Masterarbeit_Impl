from __future__ import print_function

import tensorflow as tf
from keras import backend as K

# tf.enable_eager_execution()
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


t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                 [[3, 3, 3], [4, 4, 4]],
                 [[5, 5, 5], [6, 6, 6]]])
# print(tf.slice(t, [1, 0, 0], [1, 1, 3]))

from keras import backend as b

x = tf.constant([[0., 1., 2.], [3., 4., 5.]])
print(tf.reduce_logsumexp(x))

