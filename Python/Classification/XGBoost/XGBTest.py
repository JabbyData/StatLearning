import XGBoost
import pandas as pd
import unittest

class TestXGBoost(unittest.TestCase):
    dataset = pd.read_csv("Data/Iris.csv")
    xgb_classifier = XGBoost.XGBoostClassifier()
    def test_init_pred(self):
        self.xgb_classifier.fit(self.dataset)
        self.assertEqual(self.xgb_classifier.predictions.shape,(150,))

if __name__ == "__main__":
    unittest.main()