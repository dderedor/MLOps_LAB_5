import pandas as pd
import seaborn as sns
import os

df = sns.load_dataset('diamonds')
df = df[(df[['x','y','z']] != 0).all(axis=1)]
df = df.dropna()
os.makedirs('data', exist_ok=True)
df.to_csv('data/prepared.csv', index=False)
print('Подготовленные данные сохранены в data/prepared.csv')
