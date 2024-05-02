""" Module Implementing XBG learners """
import numpy as np
class XGBNode():
    """ For now without pruning """
    def __init__(self, left_son=None, right_son=None, residuals=None, residuals_index=None, mode="C"):
        self.left_son = left_son
        self.right_son = right_son
        self.residuals = residuals
        self.residuals_index = residuals_index # to keep track of the residuals index toward predictions
        self.mode = mode

    def get_similarity_score(self, predictions):
        if self.mode == "C":
            return np.sum(self.residuals)**2 / (np.sum(predictions * (1 - predictions)))
        elif self.mode == "R":
            return np.sum(self.residuals)**2 / (np.shape(self.residuals)[0])

    def similarity_score_gain(self, left_index, right_index, predictions):
        if self.mode == "C":
            ss_l = np.sum(self.residuals[left_index])**2 / np.sum(predictions[left_index] * (1 - predictions[left_index]))
            ss_r = np.sum(self.residuals[right_index])**2 / np.sum(predictions[right_index] * (1 - predictions[right_index]))
            return ss_l + ss_r + self.get_similarity_score(predictions)

    def split(self, features, predictions):
        """ Module finding the best split for a node """
        ss_gain_best = 0
        ss_left_index_best, ss_right_index_best = [], []
        for feature in features:
            for feat_val in feature:
                left_index_res = []
                right_index_res = []
                for i in range(len(self.residuals)):
                    if feature[i] <= feat_val:
                        left_index_res.append(self.residuals_index[i])
                    else:
                        right_index_res.append(self.residuals_index[i])
            ss_gain = self.similarity_score_gain(left_index_res, right_index_res, predictions)
            if ss_gain > ss_gain_best:
                ss_left_index_best = left_index_res
                ss_right_index_best = right_index_res
                ss_gain_best = ss_gain

        return ss_left_index_best, ss_right_index_best

    def fit_rec(self, features, predictions, depth, max_depth, output_values):
        if depth < max_depth:
            left_index, right_index = self.split(features, predictions)
            self.left_son = XGBNode(residuals=self.residuals[left_index], residuals_index=left_index, mode=self.mode)
            self.right_son = XGBNode(residuals=self.residuals[right_index], residuals_index=right_index, mode=self.mode)
            output_values = self.left_son.fit_rec(features, predictions, depth + 1, max_depth, output_values)
            output_values = self.right_son.fit_rec(features, predictions, depth + 1, max_depth, output_values)
            return output_values
        else:
            # compute the output value of the current leaf
            output_value = 0
            if self.mode == "C":
                output_value = np.sum(self.residuals) / np.sum(predictions * (1 - predictions))
            elif self.mode == "R":
                output_value = np.sum(self.residuals) / np.shape(self.residuals)[0]
            output_values[self.residuals_index] = output_value
            return output_values

class XGBTree():
    def __init__(self, max_depth, mode="C"): # "C" for Classification and "R" for Regression
        self.root = None
        self.max_depth = max_depth
        self.mode = mode


    def fit(self, features, predictions, residuals):
        if self.max_depth >= 0:
            self.root = XGBNode(residuals=residuals, residuals_index=list(range(len(residuals))), mode=self.mode)
            output_values = self.root.fit_rec(features, predictions, 0, self.max_depth, np.zeros((len(residuals),)))
            return output_values

