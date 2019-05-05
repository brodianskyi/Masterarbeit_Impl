import numpy as np

sentences = ["The cat sat on the mat.",
             "The dog ate my homework"]

token_dictionary = {}

for sample in sentences:
    for word in sample.split():
        if word not in token_dictionary:
            # {"word":len++}
            token_dictionary[word] = len(token_dictionary) + 1


# length of sample
max_length = 10
# len(sentences)=2
results = np.zeros(shape=(len(sentences), max_length, max(token_dictionary.values()) + 1))

for i, sample in enumerate(sentences):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_dictionary.get(word)
        results[i, j, index] = 1


print("")