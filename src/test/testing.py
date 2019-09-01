from __future__ import print_function

import tensorflow as tf
from keras import backend as K

tf.enable_eager_execution()
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

input_shape = (2, 80, 50)
print(input_shape[:2])
print(input_shape[:2]+(5,))



