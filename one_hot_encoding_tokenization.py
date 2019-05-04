from keras.preprocessing.text import Tokenizer

samples = ["The cat sat sat sat on the mat.",
           "The cat ate my cat cat."]

# num_words - number of features
tokenizer = Tokenizer(num_words=10)
# build the dict {"word": index}
tokenizer.fit_on_texts(samples)
# based on built dict turning strings into lists of integer indices
sequences = tokenizer.texts_to_sequences(samples)
print(sequences)
# convert list of text (samples) into numpy matrix
# vectorization modes
one_hot_result = tokenizer.texts_to_matrix(samples, mode="count")
# print polye word_index in object tokenizer with the information about {"word":index}
word_index = tokenizer.word_index
# number of  unique words found = length of dict {"word": index}
print("Found %s unique tokens. " % len(word_index))