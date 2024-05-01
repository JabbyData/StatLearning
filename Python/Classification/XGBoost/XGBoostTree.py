""" Module Implementing XBG learners """

class XGBNode():
    def __init__(self, similarity_score=None, left_son=None, right_son=None):
        self.similarity_score = similarity_score
        self.left_son = left_son
        self.right_son = right_son
        # TODO : integrate residuals

class XGBTree():
    def __init__(self, max_depth, mode="C"): # "C" for Classification and "R" for Regression
        self.root = None
        self.max_depth = max_depth
        self.mode = mode

    def fit(self, features, predictions, residuals, depth):
        if depth < self.max_depth:
            pass
