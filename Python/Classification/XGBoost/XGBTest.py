import XGBoost
import pandas as pd
import unittest
import numpy as np

class TestXGBoost(unittest.TestCase):
    dataset = pd.read_csv("Data/Iris.csv")
    dataset.drop("Id", axis=1, inplace=True)
    # simplify the dataset
    dataset_binary = pd.concat([dataset[dataset['Species'] == 'Iris-setosa'],dataset[dataset['Species'] == 'Iris-versicolor']])
    dataset_binary['Species'] = dataset_binary['Species'].map(lambda x : 0 if x == 'Iris-setosa' else 1)
    xgb_classifier = XGBoost.XGBoost()
    def test_init_pred(self):
        self.xgb_classifier.fit(self.dataset_binary.iloc[:,0:-1], self.dataset_binary.iloc[:,-1])
        self.assertEqual(self.xgb_classifier.predictions.shape,(150,))

    def test_fit(self):
        features = self.dataset_binary.iloc[:,0:4]
        target = self.dataset_binary.iloc[:,4]
        assert(self.xgb_classifier.fit(features, target).shape == (100,))
        assert(self.xgb_classifier.learners != [])

    def test_score(self):
        score = self.xgb_classifier.score_train_test(self.dataset_binary)
        print("Training score : ", score[0])
        print("Testing score : ", score[1])
        assert(score.shape == (2,))

if __name__ == "__main__":
    unittest.main()