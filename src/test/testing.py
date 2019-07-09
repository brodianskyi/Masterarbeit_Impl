from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

y_true = [['1', '2', '2', "1", "2", "2"]]
y_pred = [['1', "1", "1"]]
print(f1_score(y_true, y_pred))
print(accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))



