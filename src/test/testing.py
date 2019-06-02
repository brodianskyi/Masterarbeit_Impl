from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.models import Sequential, Model, Input
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot

# define documents
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']
# define class labels
labels = ([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
# integer encode the documents
vocab_size = 50
# encoded_docs = [[13, 8],[13, 19]]
encoded_docs = [one_hot(d, vocab_size) for d in docs]
# pad documents to a max length of 4 words
max_length = 4
# padded_docs = [13 8  0 0]
#               [13 19 0 0]
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define the model
model = Sequential()
# Embedding(50, 8, 4)
# shape(None, 4, 8)
input_tensor_shape = Input(shape=(max_length, ))
model = Embedding(input_dim=vocab_size, output_dim=8, input_length=max_length)(input_tensor_shape)
model.add(Flatten())
y = Dense(16, activation='sigmoid')
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy * 100))
