import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

import numpy as np

#cuts off reviews after 100 words
maxlen = 100
# training on 200 samples
training_samples = 200
# validates on 10000 samples
validation_samples = 10000
# the max capacity of dataset
max_words = 10000

imdb_dir = "/home/rody/PycharmProjects/Testing/data_source/Reviews_dataset/aclImdb"
train_dir = os.path.join(imdb_dir, "train")

labels = []
texts = []

for label_type in ["neg", "pos"]:
    # /home/pasha/Downloads/aclImdb/aclImdb/neg
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        # cut extension .txt from the right side of string to the end
        if fname[-4:] == ".txt":
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == "neg":
                labels.append(0)
            else:
                labels.append(1)


# the max capacity of dataset
tokenizer = Tokenizer(num_words=max_words)
#
tokenizer.fit_on_texts(texts)
# turning strings into lists of integer indices
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print("Found %s unique tokens." % len(word_index))
# transforms sequence into 2D Numpy array of shape(25000 ,maxlen=100)
data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
# Shape of data tensor: (25000, 100)
print("Shape of data tensor:", data.shape)
# Shape of label tensor: (25000, )
print("Shape of label tensor", labels.shape)
# create a numpy array filled from 0 to 25000
indices = np.arange(data.shape[0])
# peremeschali indices
np.random.shuffle(indices)

# prerovnali peremeshaniye indexy
# peremeshali data i label
data = data[indices]
labels = labels[indices]

# starts from 0 up to training_sample size = 200 data[200]
x_train = data[:training_samples]
y_train = labels[:training_samples]
# data[from 200 to the end size]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]


# parse globe to build words as string to their vector representation as number vectors
glove_dir = "/home/rody/PycharmProjects/Testing/data_source/glove.6B"

embeddings_index = {}
f = open(os.path.join(glove_dir, "glove.6B.100d.txt"))

for line in f:
    # line - "pasha: 33, 22, 44"
    values = line.split()
    # word = pasha
    word = values[0]
    # coefs = 33, 22, 44
    coefs = np.asarray(values[1:], dtype="float32")
    # embeddings_index[pasha] = 33, 22, 44
    embeddings_index[word] = coefs


f.close()

print("Found %s word vectors." % len(embeddings_index))

# build embedding matrix to load into an Embedding layer
# with the shape (max_words, embedding_dim)
embeddings_dim = 100
# full fill embedding matrix
embedding_matrix = np.zeros((max_words, embeddings_dim))
for word, i in word_index.items():
    # word_index - are words from positive/negative reviews dataset
    if i < max_words:
        # get vector for word from dataset
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# defining a model
'''
model = Sequential()
model.add(Embedding(max_words, embeddings_dim, input_length=maxlen))
model.add(Embedding(max_words, embeddings_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["acc"])

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))

'''
model = Sequential()
model.add(Embedding(max_words, embeddings_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()

# set pre trained weight(word embeddings) into the Embedding layer
# sets the values of the weights of the model

model.layers[0].set_weights([embedding_matrix])
# not trained - part of the model already pretrained
# pretrained parts should not be updated during training
# to avoid forgetting what they already know
model.layers[0].trainable = False

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["acc"])

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))

# model.sample_weights("pre_trained_glove_model.h5")








