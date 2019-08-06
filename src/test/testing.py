from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras import backend as K

tf.enable_eager_execution()
my_tensor = tf.constant(
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

indices = [[0, 0],
           [1, 2]]

a = tf.gather_nd(my_tensor, indices)
print(a)