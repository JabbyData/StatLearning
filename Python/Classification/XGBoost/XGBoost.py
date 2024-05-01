""" Module implementing the XGBoost algorithm """
import numpy as np

#### 2 different types of tasks : Classification and Regression ####


class XGBoostClassifier :
    """ Classification Task """
    def __init__(self, max_iter=10, learning_rate=0.3):
        self.predictions = None
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def fit(self, features, target, init_pred=0.5):
        """ Fit the XGBoost algorithm to a dataset """
        n = target.shape[0]

        # initial prediction
        self.predictions = np.full((n,), init_pred)

        # Learning Phase
        self.fit_rec(features, target)

    def fit_rec(self, features, target):
        """ Perform XGB iterations to build learners """
        if self.max_iter == 0:
            return self.predictions

        # else, learn new predictions
        residuals = target - self.predictions
        output_values = self.build_xgb_tree(features, residuals)
        self.predictions += self.learning_rate * output_values
        self.max_iter -= 1
        self.fit_rec(features, target)

    def build_xgb_tree(self,features, target):
        """ Build a tree that fit residuals and returns output values """
        pass

