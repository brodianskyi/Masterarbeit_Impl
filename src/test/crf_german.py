import pandas as pd
import csv

data = pd.read_csv("NER-de-train.tsv", names=["Sentence", "Word", "OTR_Span", "EMB_Span"], delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')
data = data.fillna(method="ffill")
data = data.head(300)
print(data)
print("Number of sentences: ", len(data.groupby(["Sentence"])))

words = list(set(data["Word"].values))
n_word = len(words)
print("Number of words in dataset =", n_word)

otr_tags = list(set(data["OTR_Span"].values))
print("otr_tags:", otr_tags)
n_otr_tags = len(otr_tags)
print("Number of otr_tags", n_otr_tags)

emb_tags = list(set(data["EMB_Span"]))
print("emb_tags:", emb_tags)
emb_tags = len(emb_tags)
print("Number of emb_tags", emb_tags)

class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["OTR_Span"].values.tolist(),
                                                           s["EMB_Span"].values.tolist())]

        self.grouped = self.data.groupby("Sentence").apply(agg_func)
        print("self.grouped:", self.grouped)
        self.sentences = [s for s in self.grouped]
        print("self.sentence: ", self.sentences)

    # return one sentence
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


getter = SentenceGetter(data)
# sent = getter.get_next()
# print("getter.get_next():", sent)

sentences = getter.sentences
print("sentences", sentences)
