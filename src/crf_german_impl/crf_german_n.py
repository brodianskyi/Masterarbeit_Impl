import csv
import numpy as np
import pandas as pd
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional
from keras.models import Model, Input
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from src.crf_german_impl.crf_n import CRF
# from src.crf_german_impl.crf_impl import CRF
from sklearn.model_selection import train_test_split
from seqeval.metrics import f1_score, classification_report
from src.crf_german_impl.crf_loss_n import crf_loss
from src.crf_german_impl.crf_accuracies_n import crf_viterbi_accuracy
from src.crf_german_impl.crf_accuracies_n import crf_marginal_accuracy

# number of examples used in each iteration
BATCH_SIZE = 250
# number of passes through entire dataset
EPOCHS = 5
# length of the subsequence
MAX_LEN = 80
# dimension of word embedding vector
EMBEDDING = 300

data = pd.read_csv("NER-de-train.tsv", names=["Word_number", "Word", "OTR_Span", "EMB_Span", "Sentence_number"],
                   delimiter="\t",
                   quoting=csv.QUOTE_NONE, encoding='utf-8')

# data = data.head(300)
print("-------data", data)
emb_tags = list(set(data["EMB_Span"]))
print("-----emb_tags", emb_tags)
otr_tags = list(set(data["OTR_Span"]))
print("-----otr_tags", otr_tags)
n_emb_tags = len(emb_tags)
print("----n_emb_tags", n_emb_tags)
n_otr_tags = len(otr_tags)
print("----n_otr_tags", n_otr_tags)


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
print("otr_tags", otr_tags)
n_otr_tags = len(otr_tags)
emb_tags = getter.get_column(2)
n_emb_tags = len(emb_tags)
print("n_emb_tags", emb_tags)

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

X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=MAX_LEN, sequences=X, padding="post", value=word2idx["PAD"])

# convert OTR_Span in sentence to list of tag indexes
y_otr = [[tag2idx[w[1]] for w in s] for s in sentences]
y_otr = pad_sequences(maxlen=MAX_LEN, sequences=y_otr, padding="post", value=tag2idx["PAD"])

y_emb = [[tag2idx[w[2]] for w in s] for s in sentences]
y_emb = pad_sequences(maxlen=MAX_LEN, sequences=y_emb, padding="post", value=tag2idx["PAD"])

# one-hot encode
# in both cases num_classes need to be equal to n_otr_tags+1
y_otr = [to_categorical(i, num_classes=n_otr_tags + 1) for i in y_otr]
y_emb = [to_categorical(i, num_classes=n_otr_tags + 1) for i in y_emb]

# y = [to_categorical(i, num_classes=n_otr_tags + 1) for i in y]

X_tr, X_te, y_tr_otr, y_te_otr, y_tr_emb, y_te_emb = train_test_split(X, y_otr, y_emb)

# X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)
# X_tr.shape, X_te.shape, np.array(y_tr_otr).shape, np.array(y_te_otr).shape,
'''
# parse embeddings to build word as string to their vector representation
word_vector_model = gensim.models.KeyedVectors.load_word2vec_format("german.model", binary=True)
embedding_index = {}
print("word_vector_model is downloaded!")
for w, v in zip(word_vector_model.wv.index2word, word_vector_model.wv.syn0):
    embedding_index[w] = v

print("Found %s word vectors." % len(embedding_index))
# build embedding matrix to load into embedding layer
# with the shape (max_words = n_word + PAD + UNK, embedding_dim)

embedding_matrix = np.zeros((n_words + 2, EMBEDDING))
for word, i in word2idx.items():
    if i < n_words + 2:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
'''
# Model definition
# input is a "tensor", that will be passed when calling other layers to produce an output
input = Input(shape=(MAX_LEN,))
# input_dim = n_word + PAD + UNK - size of the vocabulary
# and calling embedding layer with (input) produces the output tensor model
model = Embedding(input_dim=n_words + 2, output_dim=EMBEDDING, input_length=MAX_LEN, mask_zero=True)(input)
# replace value of model, because this intermediate output is not interesting to keep
model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.5))(model)
model = TimeDistributed(Dense(50, activation="relu"))(model)
# CRF Layer = n_otr_tags + PAD
# keep the two different outputs for defining the model
# crf_otr and crf_emb are called with the same input x, creating a fork
# crf_otr = CRF(n_otr_tags + 1)
# out_otr = crf_otr(model)
crf_emb = CRF(n_otr_tags + 1)
out_emb = crf_emb(model)
model = Model(input, out_emb)
# model = Model(input, [out_otr, out_emb])
# model.summary()
# set pre trained weight into embedding layer
# model.layers[1].set_weights([embedding_matrix])
# model.layers[1].trainable = False
# model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
# loss can be one for both otr and emb or a list with different loss functions for otr and emb
model.compile("rmsprop", loss=crf_loss, metrics=[crf_marginal_accuracy])
# model.compile("adam", loss=crf_loss, metrics=[crf_viterbi_accuracy])
model.summary()
history = model.fit(X_tr, np.array(y_tr_emb), batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=2)
# history = model.fit(X_tr, [np.array(y_tr_otr), np.array(y_tr_emb)], batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=2)
# new version
test_pred = model.predict(X_te, verbose=1)

"""
model = Sequential()
embeddings = Embedding(input_dim=n_words+2, output_dim=EMBEDDING, input_length=MAX_LEN, mask_zero=True)
model.add(embeddings)
crf = CRF(n_otr_tags + 1)
model.add(crf)
"""

def pred2label(pred):
    # [[[0001][0010]]]
    '''
    for i in range(len(pred)):
        p = np.argmax(pred[i], axis=-1)
        for t, pr in zip(y_te[i], p):
            t_out.append(idx2tag[t].replace("0", "O").replace("PAD", "O"))
            pre_out.append(idx2tag[pr].replace("0", "O").replace("PAD", "O"))
    '''
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("0", "O").replace("PAD", "O"))
        out.append(out_i)
    return out


pred_labels = pred2label(test_pred)
test_labels_otr = pred2label(y_te_otr)
print("F1-score: {:.1%}".format(f1_score(test_labels_otr, pred_labels)))
print(classification_report(test_labels_otr, pred_labels))



'''
# old version
y_te_tag = np.argmax(y_tr, axis=-1)
oldStdout = sys.stdout
file_out = open("output.txt", "w")

sys.stdout = file_out
sys.stdout = oldStdout
file_out.close()

test_pred = model.predict(X_te, verbose=1)


def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "0"))
        out.append(out_i)
    return out


pred_labels = pred2label(test_pred)
test_labels = pred2label(y_te)
print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
print(classification_report(test_labels, pred_labels))
# prediction for single testing sentence
test_sent = "Einstein erhielt die Lehrberechtigung an der Berliner Universität, am 1. Oktober 1917 wurde er Direktor des Kaiser-Wilhelm-Instituts für Physik und blieb in dieser Position bis 1933. Von 1923 bis 1933 war Einstein auch Mitglied des Senats der Kaiser-Wilhelm-Gesellschaft. Seine letzte Auslandsreise außerhalb der USA nach seiner Übersiedlung dorthin unternahm Einstein 1935 auf die zu Großbritannien gehörenden Bermuda-Inseln, ein Zwangsaufenthalt aus formalen Gründen, da er damals noch nicht US-Staatsbürger war."
tok_test_sent = TweetTokenizer().tokenize(test_sent)
x_test_sent = pad_sequences(maxlen=MAX_LEN, sequences=[[word2idx.get(w, 0) for w in tok_test_sent]], padding="post", value=0)
p = model.predict(np.array([x_test_sent[0]]))
p = np.argmax(p, axis=-1)
print("{:15}||{}".format("Word", "Prediction"))
print(30 * "=")
for w, pred in zip(tok_test_sent, p[0]):
    print("{:15}: {:5}".format(w, idx2tag[pred]))
'''
