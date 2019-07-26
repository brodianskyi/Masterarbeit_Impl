import tensorflow as tf
from keras import backend as K
import numpy as np
from keras import activations
import src.crf_german_impl.data_source as dt_source

my_tensor = tf.constant(
    [
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [12, 13, 14, 15]
        ]
        ,
        [
            [16, 17, 18, 19],
            [20, 21, 22, 23],
            [24, 25, 26, 27],
            [28, 29, 30, 31]
        ]
    ]

    , dtype="float32"
)



# Shape = (2,3,4)
sess = tf.InteractiveSession()
# print(sess.run(my_tensor))
# -----------K.concatenate([K.zeros_like(x[:, :offset]), x[:, :-offset]], axis=1)--------
# offset_plus=(2, 1, 4)
offset_plus = my_tensor[:, :1]
# zeros_like_shape=(2, 1, 4)
zeros_like = K.zeros_like(offset_plus)
# offset_minus_shape=(2, 2, 4)
offset_minus = my_tensor[:, :-1]

# print("--offset_plus--", sess.run(offset_plus))
# print("zeros_like", sess.run(zeros_like))
# print("offset_minus", sess.run(offset_minus))
# shift_tensor_right = K.concatenate([zeros_like, offset_minus], axis=1)
# print("shift_tensor_right", sess.run(shift_tensor_right))
# print("concat_shape", )
#print(sess.run(concatenate))

# -----------start_mask = K.cast(K.greater(mask, self.shift_right(mask)), K.floatx())---
# (my_tensor > shift_tensor_right)
# greater = K.cast(K.greater(my_tensor, shift_tensor_right), K.floatx())
# print(sess.run(greater))

# expand_tensr = K.expand_dims(my_tensor, 1)
# print((sess.run(expand_tensr)))
# print("normal", my_tensor.shape)
# print("expand", expand_tensr.shape)
# prev_target_val = K.zeros_like(my_tensor[:, 0, :])
# prev_target_val_k = K.zeros_like(prev_target_val[:, :1])
# print(sess.run(prev_target_val))
# print(prev_target_val.shape)
# print(sess.run(prev_target_val_k))
# print(prev_target_val_k.shape)
# initial_state = [prev_target_val, K.zeros_like(prev_target_val[:, :1])]
# print("---", sess.run(initial_state))

# shape(2, 1) -> [[2][5]]


# kernel = tf.constant(np.arange(1, 9, dtype=np.float32), shape=[4, 2])
# print(sess.run(kernel))

# k_product = K.dot(dt_source.X, dt_source.kernel) + dt_source.bias
# k_produkt = dt_source.bias
#print(sess.run(k_product))

k_product = dt_source.bias
print("shape_start= ", k_product.shape, " --start=left_boundary= ", sess.run(k_product))

sess.close()




