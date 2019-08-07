from __future__ import print_function

import tensorflow as tf

tf.enable_eager_execution()
my_tensor = tf.constant(
    [[1, 2, 3], [16, 17, 18]]
    , dtype="float32"
)

a = my_tensor[:, :2]

b = my_tensor[:, 1:]


print(a)
print(b)
print(a*b)
