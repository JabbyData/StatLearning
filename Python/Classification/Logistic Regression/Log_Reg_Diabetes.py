""" Performing Logistic Regression on diabetes dataset """

### Dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report


### Loading the data
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv('../Data/diabetes.csv', header=None, names=col_names, skiprows=1)
# print(pima.head())

### Target and Features
y = pima.iloc[:,-1]
X = pima.iloc[:,:-1]

### Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

### Fitting the model
model = LogisticRegression(random_state=16,max_iter=180)
model.fit(X_train, y_train)

### Making predictions
y_pred = model.predict(X_test)

### Model Evaluation
print("Basic Classifier Accuracy : ",sum(abs(1 - y_test))/len(y_test)) # Basic Classifier : always predicting 0
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

### Confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.text(0.5,257.44,'Predicted label');
plt.show()

target_names = ['without diabetes', 'with diabetes']
print(classification_report(y_test, y_pred, target_names=target_names))