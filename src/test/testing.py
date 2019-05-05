import pandas as pd

df = pd.DataFrame({'Animal': 1,
                   'Max Speed': 2,
                   "ddf": 3})

x = lambda a: a + 5


df.groupby("ddf").apply(x)
print(df)