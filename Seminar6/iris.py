import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.csv")
# data=np.array(df)
# print(data)

X = df.iloc[:, 1:5]  # Selecting columns 1 to 4 (SepalLength, SepalWidth, PetalLength, PetalWidth)
y = df.iloc[:, 5]  # Selecting the last column (species name)


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33)  # splits the data: 67% for training X_train, y_train
                                                                     # 33% for testing X_test, y_test

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)  # trains the classifier fit using the training data


pred = clf.predict(X_test)  # Uses the trained model to predict the species for test data


print(np.mean(pred == y_test)) # Compares predictions (pred) with actual labels (y_test)
                               # Computes accuracy as the proportion of correct predictions
