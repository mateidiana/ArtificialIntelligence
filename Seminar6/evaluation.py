import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn import tree


def cv(clf, features, classes, n):
    folds = np.random.choice(np.arange(n), features.shape[0], replace=True)
    acc = np.arange(n).astype("double")
    for i in range(n):
        train = features.loc[folds != i, :]
        ytrain = classes.loc[folds != i].astype("str")
        clf.fit(train, ytrain)

        val = features.loc[folds == i, :]
        yval = classes.loc[folds == i].astype("str")

        p = clf.predict(val)
        acc[i] = sum(p == yval) / len(p)
    return np.mean(acc)


dataset = pd.DataFrame(loadtxt('pima-indians-diabetes.csv', delimiter=','))
# split into input (X) and output (y) variables
X = dataset.iloc[:, 0:8]
y = dataset.iloc[:, 8]

acctree = cv(tree.DecisionTreeClassifier(), X, y, 10)
acclda = cv(LinearDiscriminantAnalysis(), X, y, 10)
accqda = cv(QuadraticDiscriminantAnalysis(), X, y, 10)
accxgb = cv(GradientBoostingClassifier(), X, y, 10)
print(acctree)
print(acclda)
print(accqda)
print(accxgb)
