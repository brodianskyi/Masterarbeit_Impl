from __future__ import print_function

import tensorflow as tf

tf.enable_eager_execution()
my_tensor = tf.constant(
    [[1, 2, 3], [16, 17, 18]]
    , dtype="float32"
)

indices = [[0, 1],
           [1, 2]]

a = tf.gather_nd(my_tensor, indices)
print(a)
