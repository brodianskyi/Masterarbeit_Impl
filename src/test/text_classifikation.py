import pandas as pd
import os
import sys
import warnings
from keras import utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras import layers


file_dir = os.getcwd()
filepath_dict = {"yelp": file_dir + "/data_source/yelp_labelled.txt"}

df_list = []

for source_name, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=["sentence", "label"], sep="\t")
    # add another header "source" to the header of data frame
    df["source"] = source_name
    df_list.append(df)

df = pd.concat(df_list)
# print(df.iloc[0])

'''
# -------------------------------------------------------
# create vocabulary of all unique words in the sentence - feature vectors
# ignore terms with the frequency lower than min_df
sentences = ["pasha likes to play pasha likes to play", "pasha pasha pasha"]
vectorizer = CountVectorizer()
vectorizer.fit(sentences)
vocab = vectorizer.vocabulary_
# print(vocab)
array_vect = vectorizer.transform(sentences).toarray()
# print(array_vect)
'''

# define baseline model

df_yelp = df[df["source"] == "yelp"]
sentences = df_yelp["sentence"].values
# label - 0,1 - positive or negative
# split into sentence(train/test) and label(train/test)
y = df_yelp["label"].values
#  test_size - the proportion of the dataset to include in the test split
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.20, random_state=1000)
# print("shape - ", sentences_train.shape)
# sentence_train - shape = (500, )

# feature vectors of training data set

vectorizer = CountVectorizer()
# Learn a vocabulary dictionary of all tokens in the raw documents.
vectorizer.fit(sentences_train)
# Transform documents to document-term matrix.
X_train = vectorizer.transform(sentences_train)
print("shape of sentence_train = ", X_train.shape)
X_test = vectorizer.transform(sentences_test)
print("shape of sentence_test = ", X_test.shape)
Testing_string = ["The chips and sals a here is amazing!",
                  "High Quality. I love this phone!."]
vectorized_testing_string = vectorizer.transform(Testing_string)
print("shape of testing string = ", vectorized_testing_string.shape)

oldStdout = sys.stdout
file_out_before = open(file_dir+"/output/out_before_training.txt", "w")
sys.stdout = file_out_before

for i in range(50):
    print("Iteration=%s, X=%s, Out=%s" % (i, X_test[i], y_test[i]))

sys.stdout = oldStdout
file_out_before.close()
# transform label into two categories - positive or negative review
# y_train = utils.to_categorical(y_train, 2)

# name of category
# category = ["positive review", "negative review"]



# using LogisticRegression based on the input feature vector
'''
with warnings.catch_warnings():
     warnings.filterwarnings("ignore")
     classifier = LogisticRegression()
     classifier.fit(X_train, y_train)
     score = classifier.score(X_test, y_test)


print("Accuracy:", score)

# Improve regression model with Keras Model
# input_dim[1] = 1714, input_dim[0] = 750 - количество входов в нейрон
# 1714 dimensions for each feature vector = количество входов в каждый нерон = weights for each feature dimension
'''

input_dim = X_train.shape[1]
print("input_dim", input_dim)
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.summary()

# learn NN
# verbose prints a log line after every batch - false because to small size of batch - will not print
# validation_data is to evaluate the loss function and model metrics at the end of each epoch
oldStdout = sys.stdout
file_out_after = open(file_dir+"/output/out_after_training.txt", "w")
sys.stdout = file_out_after

model_fit_history = model.fit(X_train, y_train,
          epochs=100,
          verbose=1,
       # validation_data=(X_test, y_test),
          batch_size=70)


'''
# evaluate to measure accuracy of the model
# returns the loss value and metrics values for the model in the test mode
print("---------------------------------------Evaluation on train dataset----------------------------")
loss, accuracy = model.evaluate(X_train, y_train, verbose=1)
print("Training Accuracy: {:.4f}".format(accuracy))
print("---------------------------------------Evaluation on test dataset----------------------------")
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print("Testing Accuracy: {:.4f}".format(accuracy))

'''

# Predict
for i in range(50):
    predictions = model.predict_classes(X_test)
    print("Iteration=%s, X=%s, Predicted=%s" % (i, X_test[i], y_test[i]))

y_new = model.predict_classes(vectorized_testing_string)
print("Sentence: %s Predicted=%s" % (Testing_string[0], y_new[0]))
print("Sentence: %s Predicted=%s" % (Testing_string[1], y_new[1]))

# Probability prediction
y_new = model.predict_proba(vectorized_testing_string)
print("Sentence: %s Probability belong to class 1 =%s" % (Testing_string[0], y_new[0]))
print("Sentence: %s Probability belong to class 1 =%s" % (Testing_string[1], y_new[1]))


sys.stdout = oldStdout
file_out_after.close()






