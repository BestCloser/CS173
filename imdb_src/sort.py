import pandas as pd
df = pd.read_csv('dataset.csv')
df = df.sort_values(by=['sentiment'])
df.to_csv('sorted.csv', index=False)