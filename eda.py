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
import pickle

data = pd.read_csv('forestfires.csv')

print("Колонки в датасете:")
print(data.columns)
print()
print("Обзор содержащейся информации:")
print(data.head())
print()

print("Посмотрим основную информацию")
print(data.describe())

print("Посмотрим распределение целевой перемнной")

plt.figure(figsize = (10, 8))
plt.title('Распределение log(area)')
plt.hist(np.log1p(data['area']))
plt.xlabel('Площадь пожара')
plt.ylabel('Частота')
plt.show()

print("Видим много нулевых значений в целевой переменной и редкие выбросы.")
print()
print("Посмотрим корреляцию целевой переменной с признаками")
print()

data_corr = data.select_dtypes(exclude = 'object').corr()
sns.heatmap(data_corr, cmap = 'coolwarm', annot = True, fmt = '.2f' )
plt.show()

"""Посмотрим на взаимосвязь
1.   Площади пожара от месяца
2.   Площади пожара от климатических изменений (ветра, температуры и тп)
3. Распределение по регионам
"""

fig, axs = plt.subplots(3, 2, figsize = (15, 20))
plt.subplots_adjust(hspace=0.5, wspace=0.4)

#1. Распредление по месяцам
x1 = data['month']
y1 = data['area']
sns.boxplot(x = x1, y = y1, ax = axs[0, 0], order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sen', 'oct', 'nov', 'dec'])
axs[0, 0].set_yscale('log')
axs[0, 0].set_title('Зависимость площади от месяца пожара')
axs[0,0].set_xlabel('Месяц')
axs[0,0].set_ylabel('Площадь в log масштабе')

#2. Зависимость от дождя
x2 = data['rain']
y2 = data['area']
axs[0, 1].scatter(x2, y2)
axs[0, 1].set_yscale('log')
axs[0, 1].set_title('Зависимость площади от интенсивности дождя')
axs[0,1].set_xlabel('Дождь')
axs[0,1].set_ylabel('Площадь в log масштабе')

#3. Зависимость от температуры
x3 = data['temp']
y3 = data['area']
sns.regplot(x = x3, y = y3, ax = axs[1, 0])
axs[1, 0].set_yscale('log')
axs[1, 0].set_title('Зависимость площади от температуры')
axs[1,0].set_xlabel('Температура')
axs[1,0].set_ylabel('Площадь в log масштабе')

#4. Зависимость от относительной влажности
x4 = data['RH']
y4 = data['area']
sns.regplot(x = x4, y = y4, ax = axs[1, 1])
axs[1, 1].set_yscale('log')
axs[1, 1].set_title('Зависимость площади от влажности')
axs[1,1].set_xlabel('Относительная влажность')
axs[1,1].set_ylabel('Площадь в log масштабе')

#5. Зависимось от ветра
x5 = data['wind']
y5 = data['area']
sns.regplot(x = x5, y = y5, ax = axs[2, 0])
axs[2, 0].set_yscale('log')
axs[2, 0].set_title('Зависимость площади пожара от скорости ветра')
axs[2,0].set_xlabel('Скорость ветра')
axs[2,0].set_ylabel('Площадь в log масштабе')

#6. Карты рисков: зависимость пожара от региона
x6 = data['X']
y6 = data['Y']
sns.kdeplot(x = x6, y = y6, ax= axs[2, 1], fill = True, alpha = 0.7)
sns.scatterplot(x = x6, y = y6, hue = np.log1p(data['area']), ax = axs[2, 1])
axs[2, 1].set_title('Карта рисков: площадь пожара от региона')
axs[2,1].set_xlabel('Координата Х')
axs[2,1].set_ylabel('Координата У')

plt.tight_layout()
plt.show()

"""**Выводы:**
1. Видно, что в мае и декабре самые большие по площади пожары. При этом в мае широкий разброс целевой переменной. Также стоит заметить, что в августе больше всего выбросов
2. Пожары возникают, когда нет дождя (за исключением нескольких выбросов)
3. Видна динамика: чем выше температура, тем чаще возникают пожары и тем больше их площадь
4. Аналогично, чем выше относительная влажность, тем реже и с меньшей площадью возникают лесные пожары
5. Скорость ветра слабо коррелирует с целевой переменной
6. Чаще всего пожары возникают в регионе с координатами X от 2 до 4, Y = 4, при этом площадь пожара небольшая
"""
print("Пропущенных значений нет")
print(data.sum().isna())
print()
print("Категориальных переменных всего 2 - это month и day. Закодируем их с помощью OneHotEncoder.")
print(data.dtypes)
print()