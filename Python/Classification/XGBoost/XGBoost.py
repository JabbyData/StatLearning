""" Module implementing the XGBoost algorithm """
import numpy as np
import XGBoostTree
from sklearn.model_selection import train_test_split

#### 2 different types of tasks : Classification and Regression ####

class XGBoost:
    """ XGBoost for Regression and Binary Classification """
    def __init__(self, max_depth=3, max_iter=10, learning_rate=0.3, min_samples_split=2):
        self.predictions = None
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.learners = []

    def fit(self, features, target, init_pred=0.5):
        """ Fit the XGBoost algorithm to a dataset """
        n = target.shape[0]

        # initial prediction
        self.predictions = np.full((n,), init_pred)

        # Learning Phase
        output_values = self.fit_rec(features, target)
        return output_values

    def fit_rec(self, features, target):
        """ Perform XGB iterations to build learners """
        if self.max_iter == 0:
            return self.predictions

        # else, learn new predictions
        residuals = target - self.predictions
        output_values = self.build_xgb_tree(features, residuals)
        self.predictions += self.learning_rate * output_values
        self.max_iter -= 1
        return self.fit_rec(features, target)

    def build_xgb_tree(self,features, residuals):
        """ Build a tree that fit residuals and returns output values """
        xgb_tree = XGBoostTree.XGBTree(self.max_depth, self.min_samples_split)
        output_values = xgb_tree.fit(features, self.predictions, residuals)
        self.learners.append(xgb_tree)
        return output_values

    def predict(self, features):
        preds = np.zeros((features.shape[0],))
        for tree in self.learners:
            output_values = tree.predict(features, preds)
            preds += self.learning_rate * output_values
        return preds.map(round)

    def score_train_test(self,dataset, random_state=42, test_size=0.2):
        """ compute the score of the model on a dataset """
        X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:,0:-1], dataset.iloc[:,-1], test_size=test_size, random_state=random_state)
        index_train = range(len(y_train))
        index_test = range(len(y_test))
        X_train.index = index_train
        y_train.index = index_train
        X_test.index = index_test
        y_test.index = index_test
        train_preds = self.fit(X_train, y_train)
        return np.mean(train_preds == y_train), np.mean(self.predict(X_test) == y_test)