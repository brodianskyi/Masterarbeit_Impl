import csv

import pandas as pd

data = pd.read_csv("NER-de-train.tsv", names=["Word_number", "Word", "OTR_Span", "EMB_Span", "Sentence_number"],
                   delimiter="\t",
                   quoting=csv.QUOTE_NONE, encoding='utf-8')
data = data.fillna(method="ffill")
data = data.head(300)
# print(data)
# print("Number of sentences: ", len(data.groupby(["Sentence"])))

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

    def prepare_data(self):
        for i in self.data["Sentence"].values:
            if i == self.sentence_begin:
                print("Beginning of the sentence")

    def __init__(self, data):
        self.sentence_info = list()
        self.sentences = list()
        self.main_list = list()
        self.n_sent = 1
        self.data = data
        self.empty = False
        sentence_count = 0
        step_count = -1

        for i in self.data["Word_number"].values:
            # print("i = ", i)
            step_count += 1
            if i == "#":
                # print("--------------sentence begin")
                sentence_count += 1
                self.data.at[step_count, "Sentence_number"] = sentence_count
                # print("Sentence: {}".format(sentence_count))
                self.sentence_info.append((self.data.at[step_count, "Word"],
                                           self.data.at[step_count, "OTR_Span"]))
            else:
                # print("step_count=", step_count)
                # print("ppp", self.data.at[step_count, "Word"])
                self.data.at[step_count, "Sentence_number"] = sentence_count
                self.sentences.append((self.data.at[step_count, "Word"],
                                       self.data.at[step_count, "OTR_Span"],
                                       self.data.at[step_count, "EMB_Span"]))
            # print(self.sentence_info)
            # print(self.grouped)

        self.main_list = [s for s in self.data.groupby("Sentence_number")]
        print(self.main_list)


getter = SentenceGetter(data)
