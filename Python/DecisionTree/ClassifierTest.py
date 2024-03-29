import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Python.DecisionTree import Bagging, BasicClassifierTree
from sklearn.ensemble import RandomForestClassifier
from AdaBoost import AdaBoostClassifier as ABC

col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
data = pd.read_csv("../../Data/iris/Iris.csv", skiprows=1, header=None, names=col_names)
mapping = {'Iris-setosa': -1, 'Iris-versicolor': 0, 'Iris-virginica': 1}
data['type'] = data['type'].map(mapping)

# CT = BasicClassifierTree.DecisionTreeClassifier(3, 2)
# CT.root = CT.build_tree(np.array(data),0)
# CT.display(data.columns.tolist())

# Train Test Split
# X_train,X_test,Y_train,Y_test = train_test_split(data.iloc[:, :-1].values,data.iloc[:,-1].values.reshape(-1,1),test_size=0.2)
# CT.fit(X_train,Y_train)
# Y_pred = CT.predict(X_test)
# print("Accuracy using basic classifier :",accuracy_score(Y_test, Y_pred))
#
# # Testing Bagging
# bagging_preds, bagging_score = Bagging.Bagging(CT, data, 10)
# print("Accuracy using bagging : ",bagging_score)
#
# # Performing Random Forest
# rfc = RandomForestClassifier(n_estimators=100)
# Y_train = np.ravel(Y_train) # format conversion
# # print(Y_train.shape)
# rfc.fit(X_train,Y_train)
# Y_pred = rfc.predict(X_test)
# print("Accuracy using Random Forest : ",accuracy_score(Y_test, Y_pred)) # a little bit less efficient than the original model, maybe because we are in low dimensions

# AdaBoost Test
X,y = data.iloc[:,:-1].values, data.iloc[:,-1].values.reshape(-1,1)
abc = ABC()
stumps, asays = abc.fit(X,y,3)
# print(stumps)
# print(asays)
# # Prediction test
# print(round(abc.make_prediction(X[3])))
print("Accuracy with AdaBoost : ",abc.score_accuracy(X,y))