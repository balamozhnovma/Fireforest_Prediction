import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_error

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score

import mlflow

data = pd.read_csv('forestfires.csv')

ohe = OneHotEncoder()
codes = ohe.fit_transform(data[['month', 'day']]).toarray()
feature_names = ohe.get_feature_names_out()
data = pd.concat([data.drop(labels = ['month', 'day'], axis = 1), pd.DataFrame(codes, columns = feature_names)], axis = 1)

X = data.drop('area', axis = 1)
y = data['area']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

q1 = data['area'].quantile(0.25)
q3 = data['area'].quantile(0.75)
iqr = q3 - q1
data_2 = data[(data['area'] < q3 + 1.5 * iqr) & (data['area'] > q1 - 1.5 * iqr)]

X_2 = data_2.drop('area', axis = 1)
y_2 = data_2['area']

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size = 0.2, random_state = 42)

pd.concat([X_train, y_train], axis  = 1).to_csv('train.csv', index = False)
pd.concat([X_test, y_test], axis  = 1).to_csv('test.csv', index = False)
pd.concat([X_train_2, y_train_2], axis = 1).to_csv('train_filtered.csv', index = False)
pd.concat([X_test_2, y_test_2], axis = 1).to_csv('test_filtered.csv', index = False)