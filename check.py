import pandas as pd

df = pd.read_csv("submission.csv")
print(df.columns)
print(len(df))