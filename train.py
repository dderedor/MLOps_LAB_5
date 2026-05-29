import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

df = pd.read_csv('data/features.csv')
X = df.drop(['price','log_price'], axis=1)
y = df['log_price']
model = LinearRegression()
model.fit(X, y)
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')
print('Модель сохранена в models/model.pkl')
