import pandas as pd

# from sklearn_crfsuite import CRF
# from sklearn_crfsuite.metrics import flat_classification_report
# from sklearn.model_selection import cross_val_predict


data = pd.read_csv("ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill")
data = data.head(400)
words = list(set(data["Word"].values))
n_word = len(words)
print(data)


# print("Number of words in dataset = ", n_word)


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]

        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        print("self.grouped: ", self.grouped)
        self.sentences = [s for s in self.grouped]

    # print("self.sentence: ", self.sentences)

    # assign a number of the sentence
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            print("first sentence ", "Sentence: {}".format(self.n_sent))
            self.n_sent += 1
            print("self.n_sent = ", self.n_sent)
            return s
        except:
            return None


getter = SentenceGetter(data)
sent = getter.get_next()
print("getter.get_next():", sent)

sentences = getter.sentences


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3]": word[-3:],
        "word[-2]": word[-2:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "postag": postag,
        "postag[:2]": postag[:2],
    }

    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            "-1:word.lower()": word1.lower(),
            "-1:word.istitle()": word1.istitle(),
            "-1:word.isupper()": word1.isupper(),
            "-1:postag": postag1,
            "-1:postag[:2]": postag1[:2],
        })
    else:
        features["BOS"] = True
    # end of the sentence
    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            "+1:word.lower()": word1.lower(),
            "+1:word.istitle()": word1.istitle(),
            "+1:word.isupper()": word.isupper(),
            "+1:postag": postag1,
            "+1:postag[:2]": postag[:2],
        })
    else:
        features["EOS"] = True
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


# token:'British', postag: 'JJ', label: 'B-gpe'
def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]

# print("Xxxx: ", X)
# print("YYY", y)

'''
crf = CRF(algorithm="lbfgs",
          c1=0.1,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)

pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)
report = flat_classification_report(y_pred=pred, y_true=y)
print(report)

crf.fit(X, y)
'''
