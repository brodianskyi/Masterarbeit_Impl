import tensorflow as tf
from keras import backend as K

my_tensor = tf.constant(
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
print("Shape=", my_tensor.shape)
# Shape = (2,3,4)
sess = tf.InteractiveSession()
# offset_plus=(2, 1, 4)
offset_plus = my_tensor[:, :1]
# zeros_like_shape=(2, 1, 4)
zeros_like = K.zeros_like(offset_plus)
# offset_minus_shape=(2, 2, 4)
offset_minus = my_tensor[:, :-1]

print("--offset_plus--", sess.run(offset_plus))
print("zeros_like", sess.run(zeros_like))
print("offset_minus", sess.run(offset_minus))
concat = K.concatenate([zeros_like, offset_minus], axis=1)
print("concat", sess.run(concat))
#concatenate = K.concatenate([zeros_like, offset_minus], axis=0)
#print(sess.run(concatenate))
sess.close()

# K.concatenate([K.zeros_like(x[:, :offset]), x[:, :-offset]], axis=1)

