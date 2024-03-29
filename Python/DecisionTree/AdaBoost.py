""" Module to implement AdaBoost algorithm """

from BasicClassifierTree import DecisionTreeClassifier as dtc
import numpy as np
import pandas as pd
import random as rd
class AdaBoostClassifier():
    def __init__(self):
        self.stumps = []
        self.amounts_of_say = []
        self.weights = []

    def fit(self,X,y,n_estimators):
        n_samples = len(y)
        self.weights = [1/n_samples]*n_samples
        if n_estimators <= 0:
            return self.stumps, self.amounts_of_say
        else:
            # building the next stump
            weak_learner = dtc(min_samples_split=1,max_depth=0) # one node
            weak_learner.root = weak_learner.fit(X,y)

            # computing total error
            total_error = self.total_error(weak_learner, X, y)

            # computing the amount of say
            amount_of_say = 0.5*np.log((1-total_error)/total_error)

            # updating the weights
            for i in range(n_samples):
                if weak_learner.make_prediction(X[i],weak_learner.root) == y[i]:
                    # well classified obersvation
                    self.weights[i] *= np.exp(-amount_of_say)
                else:
                    # missclassified observation
                    self.weights[i] *= np.exp(amount_of_say)

            # normalizing weights
            self.weights /= sum(self.weights)

            # building the next dataset
            new_X,new_y = self.new_data(X,y)

            # udpating the model and iterate
            self.stumps.append(weak_learner)
            self.amounts_of_say.append(amount_of_say)
            return self.fit(new_X,new_y,n_estimators-1)

    def total_error(self,weak_learner,X,y):
        total_error = 0
        for i in range(len(y)):
            if weak_learner.make_prediction(X[i],weak_learner.root) != y[i]:
                total_error += self.weights[i]
        return total_error

    def new_data(self,X,y):
        new_X = np.empty(X.shape)
        new_y = np.empty(y.shape)
        n = len(X)
        for i in range(len(X)):
            # BootStrapping
            rd_row = rd.uniform(0,1)
            j = 0
            w = self.weights[0]
            while j < n-1 and w < rd_row:
                j += 1
                w += self.weights[j]
            new_X[j] = X[j]
            new_y[j] = y[j]
        return new_X,new_y

    def make_prediction(self,x):
        pred = 0
        for i,stump in enumerate(self.stumps):
            pred += stump.make_prediction(x,stump.root) * self.amounts_of_say[i]
        return round(pred)

    def score_accuracy(self,X,y):
        score = 0
        for i,obs in enumerate(X):
            if self.make_prediction(obs) == y[i]:
                score += 1
        return score / len(X)