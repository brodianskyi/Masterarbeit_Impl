import csv

import pandas as pd

data = pd.read_csv("NER-de-train.tsv", names=["Word_number", "Word", "OTR_Span", "EMB_Span", "Sentence_number"],
                   delimiter="\t",
                   quoting=csv.QUOTE_NONE, encoding='utf-8')
data = data.head(300)

words = list(set(data["Word"].values))
n_word = len(words)
print("Number of words in dataset =", n_word)

otr_tags = list(set(data["OTR_Span"].values))
# print("otr_tags:", otr_tags)
n_otr_tags = len(otr_tags)
# print("Number of otr_tags", n_otr_tags)

emb_tags = list(set(data["EMB_Span"]))
# print("emb_tags:", emb_tags)
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
                self.data.at[self.step_count, "Sentence_number"] = self.sentences_count
                self.data.at[self.step_count, "EMB_Span"] = 0
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


getter = SentenceGetter(data)
sent = getter.get_first_sent()
sentences = getter.sentences
print("sentences", sentences)
