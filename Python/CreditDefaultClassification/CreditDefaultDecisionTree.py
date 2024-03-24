""" Module to compute some decision trees for the credit default classification problem """

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import warnings

# Load the data
# Prevent excessive warnings
warnings.filterwarnings("ignore", category=FutureWarning) # to prevent huge amount of warnings

# Load the data and performs the preprocessing
# Reindex the data
data = pd.read_csv("../../Data/CreditDefault/UCI_Credit_Card.csv")
data = data.drop(columns='ID') # cf natural indexing

# 10-fold cross validation to compare models
stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Create the decision trees
dtree = DecisionTreeClassifier(min_samples_leaf=1) # Basic decision tree
bagging = BaggingClassifier(base_estimator=dtree, n_estimators=100) # Bagging decision tree
random_forest = RandomForestClassifier(n_estimators=100) # Random Forest

# Perform the 10-fold cross validation
dtree_accuracies = []
bagging_accuracies = []
random_forest_accuracies = []

for train_index, test_index in stratified_kfold.split(data.iloc[:,:-1], data.iloc[:,-1]):
    X_train, X_test = data.iloc[train_index,:-1], data.iloc[test_index,:-1]
    y_train, y_test = data.iloc[train_index,-1], data.iloc[test_index,-1]

    # Decision tree
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)
    dtree_accuracies.append(accuracy_score(y_test, y_pred))

    # Bagging
    bagging.fit(X_train, y_train)
    y_pred = bagging.predict(X_test)
    bagging_accuracies.append(accuracy_score(y_test, y_pred))

    # Random Forest
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    random_forest_accuracies.append(accuracy_score(y_test, y_pred))

print("Decision Tree accuracy: ", np.mean(dtree_accuracies)) # 0.724, bof
print("Bagging accuracy: ", np.mean(bagging_accuracies)) # 0.816, better
print("Random Forest accuracy: ", np.mean(random_forest_accuracies)) # 0.818, still improving