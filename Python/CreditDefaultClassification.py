# Dependencies
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import statsmodels.api as sm

# Warning ignore
warnings.filterwarnings("ignore", category=FutureWarning) # to prevent huge amount of warnings

# Preprocessing the data
# Verifying data leakage
data = pd.read_csv("../Data/CreditDefault/UCI_Credit_Card.csv")
if not data.isnull().any().any():
    print("No data leakage found")
else:
    print("Data leakage found")

# Reindexing the data
data = data.drop(columns='ID') # no need for IDs since it the same as the index
print(data.head())

# Splitting the data + Scaling
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]

# Low dimension (25 predictors), so we are performing KNN, Log Reg, LDA and Naive Bayes.

# Discriminative models : KNN and Log Reg

# Starting with KNN : no specific assumptions
std_scaler = StandardScaler()

X_scaled = pd.DataFrame(std_scaler.fit_transform(X), columns=X.columns.tolist())

# First basic split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Finding the best value for K
# k_values = [i for i in range(20, 40)]
# scores = []
# for k in k_values:
#     print("K: ", k)
#     knn = KNeighborsClassifier(n_neighbors=k)
#     score = cross_val_score(knn, X_train, y_train, cv=10)
#     scores.append(np.mean(score))

# Finding the best K
# best_k = k_values[scores.index(max(scores))]
# print("Best K: ", best_k)

# # Plotting KNN Accuracy
# plt.plot(k_values, scores)
# plt.title("KNN Accuracy")
# plt.xlabel("K")
# plt.ylabel("Accuracy")
# plt.show()

best_k = 29

# Training the model
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Display the confusion matrix
confusion_matrix = pd.crosstab(y_test, knn.predict(X_test), rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
print("KNN Accuracy: ", knn.score(X_test, y_test))

# Performing Logistic Regression : assumes linear relationship between predictors and log odds
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
logit_fit = sm.Logit(y_train, X_train).fit()
print(logit_fit.summary())

# Display the confusion matrix
confusion_matrix = pd.crosstab(y_test, log_reg.predict(X_test), rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
print("Logistic Regression Accuracy: ", log_reg.score(X_test, y_test))

# Generative models : LDA and Naive Bayes

# Performing LDA : assumes normal distribution of predictors with the same covariance matrix for each
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Display the confusion matrix
confusion_matrix = pd.crosstab(y_test, lda.predict(X_test), rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
print("LDA Accuracy: ", lda.score(X_test, y_test))

# Performing Naive Bayes : assumes normal distribution of predictors with different covariance matrix for each
nb = GaussianNB()
nb.fit(X_train, y_train)

# Display the confusion matrix
confusion_matrix = pd.crosstab(y_test, nb.predict(X_test), rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
print("Naive Bayes Accuracy: ", nb.score(X_test, y_test))


# Step 2 : compare using 10-fold cross validation
# Model comparison using 10-fold cross validation

accuracies = {"KNN": [], "Log Reg": [], "LDA": [], "Naive Bayes": []}

for i in range(10):
    X_scaled_test,y_scaled_test = X_scaled[3000*i:3000*(i+1)],Y[3000*i:3000*(i+1)]
    X_scaled_train,y_scaled_train = pd.concat([X_scaled[0:3000*i],X_scaled[3000*(i+1):]]),pd.concat([Y[0:3000*i],Y[3000*(i+1):]])
    print("Step: ", i)

    # Select the best K for KNN
    k_values = [i for i in range(20, 40)]
    scores = []
    for k in k_values:
        print("K: ", k)
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, X_scaled_train, y_scaled_train, cv=10)
        scores.append(np.mean(score))
    best_k = k_values[scores.index(max(scores))]

    # Training models and updating scores
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_scaled_train, y_scaled_train)

    accuracies["KNN"].append(knn.score(X_scaled_test, y_scaled_test))

    log_reg = LogisticRegression()
    log_reg.fit(X_scaled_train, y_scaled_train)
    accuracies["Log Reg"].append(log_reg.score(X_scaled_test, y_scaled_test))

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_scaled_train, y_scaled_train)
    accuracies["LDA"].append(lda.score(X_scaled_test, y_scaled_test))

    nb = GaussianNB()
    nb.fit(X_scaled_train, y_scaled_train)
    accuracies["Naive Bayes"].append(nb.score(X_scaled_test, y_scaled_test))

print("KNN Accuracy: ", np.mean(accuracies["KNN"]))
print("Log Reg Accuracy: ", np.mean(accuracies["Log Reg"]))
print("LDA Accuracy: ", np.mean(accuracies["LDA"]))
print("Naive Bayes Accuracy: ", np.mean(accuracies["Naive Bayes"]))

# To sum up, KNN and LDA are the best models for this dataset for now