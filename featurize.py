import pandas as pd
import numpy as np
import os

df = pd.read_csv('data/prepared.csv')
df['volume'] = df['x'] * df['y'] * df['z']
df['log_price'] = np.log1p(df['price'])
df['density'] = df['carat'] / df['volume']
df = pd.get_dummies(df, columns=['cut','color','clarity'], drop_first=True)
os.makedirs('data', exist_ok=True)
df.to_csv('data/features.csv', index=False)
print('Признаки сохранены в data/features.csv')
