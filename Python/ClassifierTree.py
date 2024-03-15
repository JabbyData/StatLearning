# Implementation of a basic classifier
import numpy as np
import pygraphviz as pgv

class Node:
    def __init__(self, info_gain=None, feature_index=None, feature_threshold=None, left_son = None, right_son=None, value=None):

        # Internal node
        self.info_gain = info_gain
        self.feature_index = feature_index
        self.feature_threshold = feature_threshold
        self.left_son = left_son
        self.right_son = right_son

        # Terminal node
        self.value = value


class DecisionTreeClassifier:
    def __init__(self, min_samples=None, max_depth=None):
        self.root = None
        self.min_samples = min_samples
        self.max_depth = max_depth

    def build_tree(self, dataset=None, depth=None, method="gini"):
        """Recursive building of the classifier tree"""
        X,Y = dataset[:,:-1],dataset[:,-1]
        nb_samples, nb_features = np.shape(X)
        if nb_samples >= self.min_samples and depth <= self.max_depth:
            # Find the best split
            best_split = self.best_split(dataset,nb_features,method)
            if best_split["info_gain"] > 0:
                left_son = self.build_tree(best_split["data_left"],depth+1,method)
                right_son = self.build_tree(best_split["data_right"],depth+1)
                return Node(best_split["info_gain"],best_split["feature_index"],best_split["feature_threshold"],left_son,right_son)

        # Else we are facing a leaf
        value = self.compute_leaf_value(Y)
        return Node(value=value)


    def best_split(self, dataset, nb_features, method="gini"):
        """ find the best split given a dataset"""
        best_split = {}
        max_info_gain = -float("inf")
        X = dataset[:,:-1]
        Y = dataset[:,-1]

        for feature_index in range(nb_features):
            feature = X[:,feature_index]
            thresholds = np.unique(feature) # unique value to prevent redundancy
            for possible_threshold in thresholds:
                data_left, data_right = self.split(dataset,feature_index,possible_threshold)
                if len(data_left) > 0 and len(data_right) > 0:
                    y_left, y_right = data_left[:,-1], data_right[:,-1]
                    info_gain = self.compute_info_gain(Y,y_left,y_right,method)
                    if info_gain > max_info_gain:
                        best_split["info_gain"] = info_gain
                        best_split["feature_index"] = feature_index
                        best_split["feature_threshold"] = possible_threshold
                        best_split["data_left"] = data_left
                        best_split["data_right"] = data_right
        return best_split

    def split(self,dataset,feature_index,threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right
    def compute_info_gain(self,Y,y_left,y_right,method):
        if method == "entropy":
            return self.compute_info_gain_entropy(Y,y_left,y_right)
        elif method == "gini":
            return self.compute_info_gain_gini(Y,y_left,y_right)
        else:
            return None

    def compute_info_gain_entropy(self,Y,y_left,y_right):
        n = len(Y)
        e_Y = self.entropy(Y)
        e_y_l = self.entropy(y_left)
        w_l = len(y_left)/n
        e_y_r = self.entropy(y_right)
        w_r = len(y_right)/n
        return e_Y - w_l * e_y_l - w_r * e_y_r

    def entropy(self,Y):
        n = len(Y)
        entropy = 0
        outcomes = np.unique(Y)
        for outcome in outcomes:
            p_k = Y.values_count(outcome)/n
            entropy += -p_k * np.log2(p_k)
        return entropy

    def compute_info_gain_gini(self,Y,y_left,y_right):
        n = len(Y)
        g_Y = self.gini_index(Y)
        g_y_l = self.gini_index(y_left)
        w_l = len(y_left) / n
        g_y_r = self.gini_index(y_right)
        w_r = len(y_right) / n
        return g_Y - w_l * g_y_l - w_r * g_y_r

    def gini_index(self,Y):
        n = len(Y)
        gini = 0
        counts = np.unique(Y,return_counts=True)[1]
        for c in counts:
            p_k = c / n
            gini += p_k**2
        return 1 - gini

    def compute_leaf_value(self,Y):
        Y = list(Y)
        return max(Y, key=Y.count)


    def display(self):
        G = pgv.AGraph(directed=True)
        self.display_rec(self.root,G,self.root.feature_index)
        G.layout()
        G.draw("file.png")

    def display_rec(self,node,G,parent_feature):
        if node.value is not None:
            G.add_node(node.value)
            G.add_edge(parent_feature,node.value)
        else:
            G.add_node(node.feature_index)
            self.display_rec(node.left_son, G, node.feature_index)
            self.display_rec(node.right_son, G, node.feature_index)
            if parent_feature is not None:
                G.add_edge(parent_feature,node.feature_index)
