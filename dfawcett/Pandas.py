__author__ = 'DavidFawcett'
import pandas as pd
import numpy as np

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('./train.csv', header=0)

#print(df)
# print(df.head(3))
print(df.info())
print(df.describe())