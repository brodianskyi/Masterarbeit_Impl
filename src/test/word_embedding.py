from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers.embeddings import Embedding
from keras.datasets import imdb
from keras import preprocessing

max_features = 10000
maxlen = 20

((x_train, y_train), (x_test, y_test)) = imdb.load_data(num_words=max_features)

# list of integers into a 2D integer tensor of shape(samples, maxlen)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
# x_test.shape (25000, 20)
print("x_test.shape", x_test.shape)


model = Sequential()
# vocabulary of 10000(integer encoded words)
model.add(Embedding(10000, 8, input_length=maxlen))
# output of embedded layer it is 20 vectors of 8 dimension each
# flatten in one 160-element (20*8) vector to pass on to the Dense layer
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
model.summary()
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

