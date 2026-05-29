import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

model = joblib.load('models/model.pkl')
df = pd.read_csv('data/features.csv')
X = df.drop(['price','log_price'], axis=1)
y_true = df['log_price']
y_pred = model.predict(X)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}')
with open('metrics.txt', 'w') as f:
    f.write(f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\n')
