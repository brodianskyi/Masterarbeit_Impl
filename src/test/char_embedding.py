import csv

import numpy as np
import pandas as pd
from keras.layers import Bidirectional, concatenate, SpatialDropout1D
from keras.layers import LSTM, Embedding, Dense, TimeDistributed
from keras.models import Model, Input
from keras.preprocessing.sequence import pad_sequences
from seqeval.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

# number of examples used in each iteration
# BATCH_SIZE = 512 #32
# number of passes through entire dataset
# EPOCHS = 5
# length of the subsequence
max_len = 75
# length of the char sequence
max_len_char = 10
# dimension of word embedding vector
# EMBEDDING_WORD = 300 # 20
# EMBEDDING_CHAR = 15 #10

data = pd.read_csv("NER-de-train.tsv", names=["Word_number", "Word", "OTR_Span", "EMB_Span", "Sentence_number"],
                   delimiter="\t",
                   quoting=csv.QUOTE_NONE, encoding='utf-8')

data = data.fillna(method="ffill")
emb_tags = list(set(data["EMB_Span"]))
emb_tags = len(emb_tags)


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
            self.step_count += 1
            if i == "#":
                self.sentences_count += 1
            else:
                self.data.at[self.step_count, "Sentence_number"] = self.sentences_count

            agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                               s["OTR_Span"].values.tolist(),
                                                               s["EMB_Span"].values.tolist())]

        self.grouped = self.data.groupby("Sentence_number").apply(agg_func)
        self.sentences = [s for s in self.grouped]
        self.sentences_count = 0

    def get_column(self, column_number):
        column_items = list()
        for s in self.sentences:
            for w in s:
                column_items.append(w[column_number])
        return list(set(column_items))

    def get_first_sent(self):
        try:
            s = self.sentences[self.sentences_count]
            self.sentences_count += 1
            return s
        except:
            return None


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

# vocabulary {key-index: value-tag}
tag2idx = {t: i + 1 for i, t in enumerate(otr_tags)}
tag2idx["PAD"] = 0
idx2tag = {i: w for w, i in tag2idx.items()}

X_word = [[word2idx[w[0]] for w in s] for s in sentences]
X_word = pad_sequences(maxlen=max_len, sequences=X_word, padding="post", truncating="post", value=word2idx["PAD"])

chars = set()
for w in words:
    if isinstance(w, str):
        for char in w:
            chars.add(char)

n_chars = len(chars)

char2idx = {c: i + 2 for i, c in enumerate(chars)}
char2idx["UNK"] = 1
char2idx["PAD"] = 0

X_char = []
for sentence in sentences:
    sent_seq = []
    for i in range(max_len):
        word_seq = []
        for j in range(max_len_char):
            try:
                word_seq.append(char2idx.get(sentence[i][0][j]))
            except:
                word_seq.append(char2idx.get("PAD"))
        sent_seq.append(word_seq)
    X_char.append(np.array(sent_seq))

# convert OTR_Span in sentence to list of tag indexes
y = [[tag2idx[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, value=tag2idx["PAD"], padding='post', truncating='post')
X_word_tr, X_word_te, y_tr, y_te = train_test_split(X_word, y, test_size=0.1, random_state=2018)
X_char_tr, X_char_te, _, _ = train_test_split(X_char, y, test_size=0.1, random_state=2018)

# input and embedding for words
word_in = Input(shape=(max_len,))
emb_word = Embedding(input_dim=n_words + 2, output_dim=20,
                     input_length=max_len, mask_zero=False)(word_in)

# input and embeddings for characters
char_in = Input(shape=(max_len, max_len_char,))
emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=10,
                                     input_length=max_len_char, mask_zero=False))(char_in)
# character LSTM to get word encodings by characters
char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,
                                recurrent_dropout=0.6))(emb_char)

# main LSTM
x = concatenate([emb_word, char_enc])
x = SpatialDropout1D(0.3)(x)
main_lstm = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.6))(x)
out = TimeDistributed(Dense(n_otr_tags + 1, activation="softmax"))(main_lstm)

model = Model([word_in, char_in], out)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
model.summary()

history = model.fit([X_word_tr,
                     np.array(X_char_tr).reshape((len(X_char_tr), max_len, max_len_char))],
                    np.array(y_tr).reshape(len(y_tr), max_len, 1),
                    batch_size=32, epochs=21, validation_split=0.1, verbose=1)

test_pred = model.predict([X_word_te, np.array(X_char_te).reshape((len(X_char_te),
                                                                   max_len, max_len_char))])
t_out = []
pre_out = []


def pred2label(pred):
    for i in range(len(test_pred)):
        p = np.argmax(pred[i], axis=-1)
        for t, pr in zip(y_te[i], p):
            t_out.append(idx2tag[t].replace("0", "O").replace("PAD", "O"))
            pre_out.append(idx2tag[pr].replace("0", "O").replace("PAD", "O"))


'''
i = 4
p = np.argmax(test_pred[i], axis=-1)
print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
print(30 * "=")
for w, t, pred in zip(X_word_te[i], y_te[i], p):
    if w != 0:
        print("---------", p)
        print("{:15}: {:5} {}".format(idx2word[w], idx2tag[t], idx2tag[pred]))
'''
pred2label(test_pred)

# pred2label(y_te)
print("F1-score: {:.1%}".format(f1_score(t_out, pre_out)))
print(classification_report(t_out, pre_out))
