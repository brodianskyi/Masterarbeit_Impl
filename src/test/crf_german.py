import csv
import sys

import numpy as np
import pandas as pd
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional
from keras.models import Model, Input
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras_contrib.layers import CRF
from sklearn.model_selection import train_test_split

# number of examples used in each iteration
BATCH_SIZE = 512
# number of passes through entire dataset
EPOCHS = 5
# max length of review (in words)
MAX_LEN = 80
# dimension of word embedding vector
EMBEDDING = 40

data = pd.read_csv("NER-de-train.tsv", names=["Word_number", "Word", "OTR_Span", "EMB_Span", "Sentence_number"],
                   delimiter="\t",
                   quoting=csv.QUOTE_NONE, encoding='utf-8')
# data = data.head(500)
emb_tags = list(set(data["EMB_Span"]))
emb_tags = len(emb_tags)


# print("Number of emb_tags", emb_tags)


class SentenceGetter(object):

    def __init__(self, data):
        self.sentences_info = list()
        self.sentences = list()
        self.grouped = list()
        self.data = data
        self.empty = False
        self.sentences_count = 0
        self.step_count = -1

        for i in self.data["Word_number"].values:
            # print("i = ", i)
            self.step_count += 1
            if i == "#":
                self.sentences_count += 1
                self.sentences_info.append((self.data.at[self.step_count, "Word"],
                                            self.data.at[self.step_count, "OTR_Span"]))
            else:
                self.data.at[self.step_count, "Sentence_number"] = self.sentences_count

            agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                               s["OTR_Span"].values.tolist(),
                                                               s["EMB_Span"].values.tolist())]

        self.grouped = self.data.groupby("Sentence_number").apply(agg_func)
        self.sentences = [s for s in self.grouped]
        self.sentences_count = 0

    def get_first_sent(self):
        try:
            s = self.sentences[self.sentences_count]
            self.sentences_count += 1
            return s
        except:
            return None

    def get_column(self, column_number):
        column_items = list()
        for s in sentences:
            for w in s:
                column_items.append(w[column_number])
        return list(set(column_items))


getter = SentenceGetter(data)
sent = getter.get_first_sent()
sentences = getter.sentences
words = getter.get_column(0)
n_words = len(words)
otr_tags = getter.get_column(1)
n_otr_tags = len(otr_tags)

word2idx = {w: i + 2 for i, w in enumerate(words)}
# unknown words
word2idx["UNK"] = 1
# padding
word2idx["PAD"] = 0
# vocabulary {key-index: value-word}
idx2word = {i: w for w, i in word2idx.items()}

# vocabulary {key-tag: value-index+2}
tag2idx = {t: i + 1 for i, t in enumerate(otr_tags)}
tag2idx["PAD"] = 0
# vocabulary {key-index: value-tag}
idx2tag = {i: w for w, i in tag2idx.items()}

# Sentences[3] = [("Bayern", "B-ORG", "B-LOC"), ("München", "I-ORG", "B-LOC")]
# print("The word ARD-Programmchef is identified by the index:{}".format(word2idx["ARD-Programmchef"]))
# print("The labels B-LOC(which defines locations) is identified by the index:{}".format(tag2idx["B-LOC"]))

X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=MAX_LEN, sequences=X, padding="post", value=word2idx["PAD"])

# convert OTR_Span in sentence to list of tag indexes
y = [[tag2idx[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=MAX_LEN, sequences=y, padding="post", value=tag2idx["PAD"])

# one-hot encode
y = [to_categorical(i, num_classes=n_otr_tags + 1) for i in y]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)
X_tr.shape, X_te.shape, np.array(y_tr).shape, np.array(y_te).shape

print("Raw Sample: ", " ".join([w[0] for w in sentences[0]]))
print("Raw Label: ", " ".join([w[1] for w in sentences[0]]))

# Model definition
input = Input(shape=(MAX_LEN,))
# input_dim = n_word + PAD + UNK
model = Embedding(input_dim=n_words + 2, output_dim=EMBEDDING, input_length=MAX_LEN, mask_zero=True)(input)
model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(model)
model = TimeDistributed(Dense(50, activation="relu"))(model)
# CRF Layer = n_otr_tags + PAD
crf = CRF(n_otr_tags + 1)
out = crf(model)
model = Model(input, out)
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
model.summary()
history = model.fit(X_tr, np.array(y_tr), batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=2)

y_te_tag = np.argmax(y_tr, axis=-1)
oldStdout = sys.stdout
file_out = open("output.txt", "w")
sys.stdout = file_out
for i in range(370):
    sent_loop = X_tr[i]
    p = model.predict(np.array([sent_loop]))
    p = np.argmax(p, axis=-1)
    idx_of_word = [idx2word[row] for row in sent_loop]
    pred_tag_of_word = [idx2tag[row] for row in p[0]]
    tag_of_word = [idx2tag[row] for row in y_te_tag[i]]
    print("idx_of_word----", idx_of_word)
    print("pred_tag_of_word-----", pred_tag_of_word)
    print("tag_of_word-----", tag_of_word)
    print("#########################################")

sys.stdout = oldStdout
file_out.close()
# report = flat_classification_report(y_pred=pred_tag, y_true=y_te_true_tag)
# print(report)
