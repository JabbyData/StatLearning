""" Performing SVM classification """
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random as rd

rd.seed(0)
data = pd.read_csv('../Data/diabetes.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
clf = SVC(kernel='linear',C=1,random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
clf.fit(X_train, y_train)
print("basic classifier accuracy :",sum(abs(1 - y_test))/len(y_test))
print("svm accuracy : ",clf.score(X_test, y_test))