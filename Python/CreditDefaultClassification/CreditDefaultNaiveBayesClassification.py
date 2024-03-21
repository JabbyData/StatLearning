"""Performing Naive Bayesian on the Credit Default dataset"""
import Preprocessor
import pandas as pd
import warnings
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

# Prevent excessive warnings
warnings.filterwarnings("ignore", category=FutureWarning) # to prevent huge amount of warnings

# Load the data and performs the preprocessing
# Reindex the data
data = pd.read_csv("../../Data/CreditDefault/UCI_Credit_Card.csv")
data = data.drop(columns='ID') # cf natural indexing

# 10-fold Cross validation
nb = GaussianNB()
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]
accuracies = cross_val_score(nb, X, Y, cv=10)

# Plotting the results
plt.plot(range(1, 11), accuracies)
plt.title("LDA Accuracy depending on the fold")
plt.xlabel("Fold number")
plt.ylabel("Accuracy")
plt.show()

print("Naive Bayes with original data: ", np.mean(accuracies)) # 0.38, pretty bad

# Trying to improve the accuracy

# Scaling the data
# Normalizing the data
normalized_data = Preprocessor.normalize(data,0,1)
X = normalized_data.iloc[:,:-1]
Y = normalized_data.iloc[:,-1]
print("Naive Bayes with normalized data : ",np.mean(cross_val_score(nb, X, Y, cv=10))) # almost 100%, strange ...

# Standardizing the data
standardized_data = Preprocessor.standardize(data)
X = standardized_data.iloc[:,:-1]
Y = standardized_data.iloc[:,-1]
print("Naive Bayes with standardized data : ",np.mean(cross_val_score(nb, X, Y, cv=10)) ) # almost 100%, strange ...

# To conclude : the best accuracy is obtained with scaled data for Naive Bayes, almost 100 % is strange ...