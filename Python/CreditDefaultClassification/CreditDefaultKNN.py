"""Performing KNN on the Credit Default dataset"""
import Preprocessor
import pandas as pd
import warnings
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict as dd
import matplotlib.pyplot as plt

# Prevent excessive warnings
warnings.filterwarnings("ignore", category=FutureWarning) # to prevent huge amount of warnings

# Load the data and performs the preprocessing
# Reindex the data
data = pd.read_csv("../../Data/CreditDefault/UCI_Credit_Card.csv")
data = data.drop(columns='ID') # cf natural indexing

# Check for leakage
if not data.isnull().any().any():
    print("No data leakage found")
else:
    print("Data leakage found")
# here no leakage

# Create scaled dataset for later use
#normalized_data = Preprocessor.normalize(data,0,1) # Normalize the data
#standardized_data = Preprocessor.standardize(data) # Standardize the data

# # Train-test split for 10-fold Cross validation
# accuracies = []
# ks = dd(int) # store values of k
# nrows = int(data.shape[0])
# length = int(nrows/10) # length of each fold
# for i in range(0, 10):
#     print("Step : ", i)
#     # Train-test split
#     test_index = list(range(i*length,(i+1)*length))
#     train_index = list(set(range(nrows))-set(test_index))
#     train_fold = data.iloc[train_index]
#     test_fold = data.iloc[test_index]
#     X_train = train_fold.iloc[:,:-1]
#     y_train = train_fold.iloc[:,-1]
#
#     # Finding the best value for K using once again 10-fold cross validation
#     k_values = [i for i in range(20, 40)]
#     scores = []
#     for k in k_values:
#         print("K: ", k)
#         knn = KNeighborsClassifier(n_neighbors=k)
#         score = cross_val_score(knn, X_train, y_train, cv=10)
#         scores.append(np.mean(score))
#
#     best_k = k_values[scores.index(max(scores))]
#     ks[best_k] += 1
#     print("Best K: ", best_k)
#
#     # Evaluate performances
#     X_test = test_fold.iloc[:,:-1]
#     y_test = test_fold.iloc[:,-1]
#     knn = KNeighborsClassifier(n_neighbors=best_k)
#     knn.fit(X_train, y_train)
#     accuracies.append(knn.score(X_test, y_test))
#
# # Plotting accuracies
# plt.plot(range(1,11),accuracies)
# plt.title("Influence of train-test split on KNN Accuracy")
# plt.xlabel("Split number")
# plt.ylabel("Accuracy")
# plt.show() # Accuracy depends on the fold chosen, hence cross validation is very useful
#
# print("Best K: ", max(ks, key=ks.get))
# print("Average accuracy: ", np.mean(accuracies))

# Best K found : 36
best_k = 36
# Average accuracy : 0.772

# Using the whole dataset
knn = KNeighborsClassifier(n_neighbors=best_k)
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]
print("Accuracy using original data  : ",np.mean(cross_val_score(knn, X, Y, cv=10))) # 0.7802, a good start

# Improving the accuracy
# We can improve the accuracy by scaling the data

# Normalizing the data
normalized_data = Preprocessor.normalize(data,0,1) # Normalize the data
knn = KNeighborsClassifier(n_neighbors=best_k)
X = normalized_data.iloc[:,:-1]
Y = normalized_data.iloc[:,-1]
print("Accuracy using normalized data  : ",np.mean(cross_val_score(knn, X, Y, cv=10))) # 0.9444, very good improvement

# Standardizing the data
standardized_data = Preprocessor.standardize(data) # Standardize the data
knn = KNeighborsClassifier(n_neighbors=best_k)
X = standardized_data.iloc[:,:-1]
Y = standardized_data.iloc[:,-1]
print("Accuracy using standardized data  : ",np.mean(cross_val_score(knn, X, Y, cv=10))) # 0.8989, good improvement

# To conclude, K = 36 + Normalization is the best choice