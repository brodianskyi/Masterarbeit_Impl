from sklearn.datasets.samples_generator import make_blobs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import numpy as np

X, y = make_blobs(n_samples=10, centers=10, n_features=4,
                  random_state=1)
# X - array of shape [n_samples, n_features]
# print("X", X)
# print("X.shape befor", X.shape)
# y - integer labels for cluster membership of each sample
# print("y", y)
# print("y shape", y.shape)

# print("-------String processing--------------------")
corpus_train = [
          "John likes ice ice cream",
          "John hates Kate"]

corpus_test = [
          'Papa loves mambo ',
          'Mary loves mambo ',
          'Pa pa pa paparam pa.']

vectorizer = CountVectorizer()
vectorizer.fit(corpus_train)
vocabulary = vectorizer.vocabulary_
X_train = vectorizer.transform(corpus_train)
print("X_train.shape", X_train.shape)
print("X_train", X_train)
print("X_train.toarray()", X_train.toarray())
print(vocabulary)
'''
X_test = vectorizer.transform(corpus_test)
print("X_test.shape", X_test.shape)
print("X_fit_test.toarray()", X_test)
print(vectorizer.get_feature_names())

print("X_train.shape[1] = ", X_train.shape[1])
print("X_test.shape[1] = ", X_test.shape[1])




# ----------------------------------------------------
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)
# print("X.shape after", X.shape)
# print(X)
'''





