import ClassifierTree
import pandas as pd
import numpy as np
import os

col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
data = pd.read_csv("../Data/iris/Iris.csv", skiprows=1, header=None, names=col_names)
print(data.head(10))

CT = ClassifierTree.DecisionTreeClassifier(5,4)
CT.root = CT.build_tree(np.array(data),0)
CT.display()