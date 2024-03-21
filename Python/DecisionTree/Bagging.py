""" Module to perform bagging and improve our simple model """
import BasicClassifierTree
import numpy as np
from sklearn.metrics import accuracy_score
def Bagging(decisionTree,dataset,B):
    """ Create bootstrapped datasets and averages predictions (majority vote) for classification """
    n = len(dataset)
    avg_preds = [0 for _ in range(n)]
    avg_score = 0
    for b in range(B):
        # bootstrap takes 2/3 of the dataset on average, hence creating a natural training dataset
        bt_train_indexes = np.random.choice(range(n),n,replace=True).tolist()
        bt_train = dataset.iloc[bt_train_indexes]
        bt_test_indexes = dataset.index.difference(bt_train_indexes)[:-1]
        bt_test = dataset.iloc[bt_test_indexes]

        # fitting the model
        X_train, Y_train = bt_train.iloc[:, :-1].values, bt_train.iloc[:, -1].values.reshape(-1,1)
        decisionTree.fit(X_train,Y_train)

        # predicting results
        X_test,Y_test = bt_test.iloc[:, :-1].values,bt_test.iloc[:,-1].values
        y_preds = decisionTree.predict(X_test)

        # updating coefficients
        for i in range(len(bt_test_indexes)):
            avg_preds[bt_test_indexes[i]] += y_preds[i]

        avg_score += accuracy_score(y_preds,Y_test)

    # Final results
    return list(map(lambda x: x / B, avg_preds)),avg_score/B