import keras.src.layers
import numpy as np
import pandas as pd

from keras import layers, Sequential
from keras import Input
from keras.src.layers import Dense, Dropout

from sklearn import preprocessing

import matplotlib.pyplot as plt
from matplotlib import gridspec

from IPython.display import Markdown
import IPython

raw = pd.read_csv("Regression_BSD_hour.csv")

X = pd.DataFrame.copy(raw)

X_days = X['dteday']
y = X['cnt']

all_days = len(X) // 24
print("Total observations", len(X))
print("Total number of days", all_days)

# Part the data into training data (70% of all days) and testing data (30%)
days_for_training = int(all_days * 0.7)
hours_for_training = days_for_training * 24
X_train = X[0:hours_for_training]
X_test = X[hours_for_training:]

# Delete specific columns
del X_train['dteday']
del X_test['dteday']
del X_train['cnt']
del X_test['cnt']

del X_train['instant']
del X_test['instant']
del X_train['casual']
del X_test['casual']
del X_train['registered']
del X_test['registered']

print("Observations for training", X_train.shape)  # 12144 observations, 12 columns
print("Observations for testing", X_test.shape)  # 5235 observations, 12 columns

y_train = y[:hours_for_training]
y_test = y[hours_for_training:]

print("Observations and targets for training", X_train.shape, y_train.shape)
print("Observations and targets for testing", X_test.shape, y_test.shape)

features = X_train.shape[1]  # 12, meaning the model expects 12 input features per sample
model = Sequential()  # layers are stacked sequentially
model.add(Input(shape=(features,)))  # the model expects input data with 12 features

# fully connected layer with 20 neurons.
# uses ReLU activation, which helps in learning complex patterns
# names the layer "Hidden1" for easier reference
model.add(Dense(20, activation='relu', name='Hidden1'))

model.add(Dropout(0.25))  # Randomly drops 25% of neurons during training to prevent over fitting

# single neuron for output (useful for regression problems)
# no activation function (output is a continuous value)
model.add(Dense(1, activation='linear', name='Output'))

# uses the Adam optimizer, which adapts learning rates automatically
# since this is a regression task, Mean Squared Error (MSE) is used as the loss function
model.compile(optimizer='adam', loss="mse")

model.summary()

print('Model Input shape:  ', model.input_shape)
print('Model Output shape: ', model.output_shape)

# trains the neural network using the fit() method
# the model goes through the entire training dataset 1000 times
# the model processes 1024 samples per batch before updating weights
# displays a progress bar with loss/accuracy metrics per epoch
results = model.fit(
    X_train,
    y_train,
    epochs=1000,
    validation_data=(X_test, y_test),
    batch_size=1024,
    verbose=1)

# plots the training history of the model to visualize training loss and validation loss per epoch
pd.DataFrame.from_dict(results.history).plot(figsize=(10, 7))
plt.show()

# evaluates the trained model on X_train and compares predictions with actual values (y_train)
X_eval = X_train
y_eval = y_train

y_pred = model.predict(X_eval).flatten()

prediction = pd.Series(y_pred, index=X_eval.index)

pd.DataFrame.from_dict({"y_eval": y_eval, "y_pred": y_pred}).head(10)


# visualizes time-series data (e.g., bike rentals per hour) along with
# input variables and model predictions. It helps analyze how well the model fits the data
#  X=input features, y=target variable (number of rentals per hour), X_days=date labels for x-axis (list of timestamps)
def plot_data(X, y, X_days, first_day=3 * 7, duration_days=3 * 7, prediction=None):
    s = first_day * 24  # start hour (e.g., 21 days * 24 hours = 504th hour)
    e = s + duration_days * 24  # end hour  (e.g., 42 days * 24 hours = 1008th hour)

    #  If prediction is not provided → Create 2 subplots ax0 → Plots input features (e.g., temperature, humidity).
    # ax1 → Plots actual rental count.

    # If prediction is provided → Create 3 subplots:
    # ax0 → Input features.
    # ax1 → Actual vs. predicted rentals.
    # ax2 → Residuals (errors = actual - predicted).
    if prediction is None:
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
        ax2 = None
    else:
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(18, 12), sharex=True,
                                            gridspec_kw={'height_ratios': [3, 3, 1]})

    # Adds vertical grey lines for working days
    for x, v in X['workingday'][s:e].items():
        if v == 1:
            ax0.axvline(x, lw=3, c='lightgrey')
            ax1.axvline(x, lw=3, c='lightgrey')
            if ax2: ax2.axvline(x, lw=3, c='lightgrey')

    # Adds dashed lines at midnight for each day.
    # Stores midday (noon) timestamps for better x-axis labeling.
    mid_day_indexes = []

    for x, v in X['hr'][s:e].items():
        if v == 0:
            ax0.axvline(x, ls=':', c='grey')
            ax1.axvline(x, ls=':', c='grey')
            if ax2: ax2.axvline(x, ls=':', c='grey')
        if v == 12:
            mid_day_indexes.append(x)

    # Plots weather-related input features (e.g., temperature, humidity, wind speed)
    for c in [
        'temp', 'hum', 'windspeed', 'weathersit',
        # 'atemp', 'season', 'workingday', 'instant', 'dteday', 'weekday',
        # 'yr', 'mnth', 'hr', 'holiday', 'casual', 'registered',  'cnt'
    ]: ax0.plot(X[c][s:e], label=c)

    ax0.legend(loc="upper left")
    ax0.set_ylabel('Input variables')

    # Red dotted line → Actual values (y).
    # Blue dotted line (if available) → Model predictions.
    ax1.plot(y[s:e], 'r:', label="ground truth")

    if prediction is not None:
        ax1.plot(prediction[s:e], 'b:', label="prediction")

    ax1.legend(loc="upper left")
    ax1.set_ylabel('Number of Rentals per hour')

    # Residuals = Actual - Predicted → Helps diagnose model accuracy.
    # High residuals indicate poor predictions.
    # Uses mid-day timestamps to label x-axis.
    if ax2:
        ax2.plot(y[s:e] - prediction[s:e], label="residuals (GT-Pred)")
        ax2.set_ylabel('Residuals')
        ax2.legend(loc="upper left")

        ax2.set_xticks(mid_day_indexes)
        ax2.xaxis.set_ticklabels([X_days[i] for i in mid_day_indexes], rotation=90)
    else:
        ax1.set_xticks(mid_day_indexes)
        ax1.xaxis.set_ticklabels([X_days[i] for i in mid_day_indexes], rotation=90)

    plt.tight_layout()
    plt.show()


# Plots 21 days of training data, model predictions, and residuals.
plot_data(X_train, y_train, X_days, prediction=prediction)
