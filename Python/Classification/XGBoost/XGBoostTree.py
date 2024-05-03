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
        self.learning_step = []
        self.output_value = 0
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

    def split(self, features, predictions, min_samples_split):
        """ Module finding the best split for a node """
        ss_gain_best = 0
        ss_left_index_best, ss_right_index_best = [], []
        feat_best, threshold_best = None, None
        for feature in features.columns:
            for feat_val in features[feature]:
                left_index_res = []
                right_index_res = []
                for i in range(len(self.residuals)):
                    if features[feature][i] <= feat_val:
                        left_index_res.append(self.residuals_index[i])
                    else:
                        right_index_res.append(self.residuals_index[i])
            if len(left_index_res) >= min_samples_split and len(right_index_res) >= min_samples_split:
                ss_gain = self.similarity_score_gain(left_index_res, right_index_res, predictions)
                if ss_gain > ss_gain_best:
                    ss_left_index_best = left_index_res
                    ss_right_index_best = right_index_res
                    ss_gain_best = ss_gain
                    feat_best = feature
                    threshold_best = feat_val

        return ss_left_index_best, ss_right_index_best, feat_best, threshold_best

    def fit_rec(self, features, predictions, depth, max_depth, output_values, min_samples_split):
        if depth < max_depth:
            left_index, right_index, feat_split, thresold_split = self.split(features, predictions, min_samples_split)
            self.learning_step = [feat_split, thresold_split]
            self.left_son = XGBNode(residuals=self.residuals[left_index], residuals_index=left_index, mode=self.mode)
            self.right_son = XGBNode(residuals=self.residuals[right_index], residuals_index=right_index, mode=self.mode)
            output_values = self.left_son.fit_rec(features, predictions, depth + 1, max_depth, output_values, min_samples_split)
            output_values = self.right_son.fit_rec(features, predictions, depth + 1, max_depth, output_values, min_samples_split)
            return output_values
        else:
            # compute the output value of the current leaf
            output_value = 0
            if self.mode == "C":
                output_value = np.sum(self.residuals) / np.sum(predictions * (1 - predictions))
            elif self.mode == "R":
                output_value = np.sum(self.residuals) / np.shape(self.residuals)[0]
            output_values[self.residuals_index] = output_value
            self.output_value = output_value
            return output_values

class XGBTree():
    def __init__(self, max_depth, min_samples_split, mode="C"): # "C" for Classification and "R" for Regression
        self.root = None
        self.max_depth = max_depth
        self.mode = mode
        self.min_samples_split = min_samples_split

    def fit(self, features, predictions, residuals):
        if self.max_depth >= 0:
            self.root = XGBNode(residuals=residuals, residuals_index=list(range(len(residuals))), mode=self.mode)
            output_values = self.root.fit_rec(features, predictions, 0, self.max_depth, np.zeros((len(residuals),)), self.min_samples_split)
            return output_values

    def predict(self, features, predictions):
        output_values = np.zeros((features.shape[0],))
        output_index = np.arange(features.shape[0])
        self.fill_output_values(self.root, features, output_values, output_index)
        return output_values

    def fill_output_values(self, node, features, output_values, output_index):
        if node.learning_step != []:
            left_index = []
            right_index = []
            for i in output_index:
                if features[node.learning_step[0]][i] <= node.learning_step[1]:
                    left_index.append(i)
                else:
                    right_index.append(i)
            self.fill_output_values(node.left_son, features, output_values, left_index)
            self.fill_output_values(node.right_son, features, output_values, right_index)
        else:
            output_values[output_index] = node.output_value
            return output_values


