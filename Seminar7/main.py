import keras.src.layers
import numpy as np
import pandas as pd

from keras import layers, Sequential
from keras import Input

from sklearn import preprocessing

import matplotlib.pyplot as plt
from matplotlib import gridspec

from IPython.display import Markdown


raw = pd.read_csv("Regression_BSD_hour.csv")
X = pd.DataFrame.copy(raw)

X_days = X['dteday']
y = X['cnt']

all_days = len(X) // 24
print("Total observations", len(X))
print("Total number of days", all_days)
days_for_training = int(all_days * 0.7)
hours_for_training = days_for_training*24
X_train = X[0:hours_for_training]
X_test = X[hours_for_training:]

del X_train['dteday']
del X_test['dteday']
del X_train['cnt']
del X_test['cnt']

print("Observations for training", X_train.shape)
print("Observations for testing", X_test.shape)

print(X_train.columns)

y_train = y[:hours_for_training]
y_test = y[hours_for_training:]

print("Observations and targets for training", X_train.shape, y_train.shape)
print("Observations and targets for testing", X_test.shape, y_test.shape)
