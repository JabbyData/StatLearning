import pandas as pd
import math as m
import graphviz
import os
import random as rd

def value_obs(observation):
    return (observation == "P").sum()/len(observation)
def entropy(observation):
    p = value_obs(observation)
    if p == 0 or p == 1:
        return 0
    return - p * m.log(p,2) - (1-p) * m.log(1-p,2)
class Node:
    def __init__(self, entropy=0, value=None, left_son=None, right_son=None, name=None, edge_label=None, sub_entropy=0, nb_leaves=1):
        self.entropy = entropy
        self.value = value
        self.name = name
        self.left_son = left_son
        self.right_son = right_son
        self.edge_label = edge_label # for graphviz, label on the edge with its parent
        self.sub_entropy = sub_entropy
        self.nb_leaves = nb_leaves

    def copy(self):
        if self.nb_leaves == 1:
            return Node(self.e,self.value,None,None,self.name,self.edge_label,self.sub_entropy,self.nb_leaves)
        else:
            l,r = self.copy(self.left_son), self.copy(self.right_son)
            return Node(self.e,self.value,l,r,self.name,self.edge_label,self.sub_entropy,self.nb_leaves)

    def copy_until(self,i,c):
        """ copy all the tree except the subnode with root at index i """
        if c == i or self.nb_leaves == 1:
            return Node(self.e,self.value,None,None,self.name,self.edge_label,self.sub_entropy,self.nb_leaves)






class ClassifierTree:
    def __init__(self, min_obs=2, max_depth=None):
        self.root = None
        self.min_obs = min_obs
        self.max_depth = max_depth

    def build_tree(self, dataset):
        x, y = dataset.iloc[:,:-1], dataset.iloc[:,-1]
        self.root = self.rec_build_tree(x,y,0)

    def rec_build_tree(self,x,y,depth,label=None):
        p = value_obs(y)
        e = entropy(y)
        # Stopping criterions : Too deep, too few obs or pure/impure
        if depth >= self.max_depth or len(y) <= self.min_obs or p == 1 or p == 0:
            return Node(e,p,None,None,"Leaf",label,e,1)
        # Else binary split to maximize info gain
        max_info = -float("inf")
        index = None
        for feature_index in x.columns:
            if len(x[feature_index].unique()) == 2:
                res_l, res_r = x[feature_index].unique() # 2 values possible to begin with
                y_l = y[x[feature_index] == res_l] # boolean indexing to dispatch the dataset
                y_r = y[x[feature_index] == res_r]
                # Calculate info gain and select the best dispatch
                info_gain = e - entropy(y_l) - entropy(y_r)
                if info_gain > max_info:
                    max_info = info_gain
                    index = feature_index
        if index != None:
            res_l, res_r = x[index].unique()
            x_l, x_r = x[x[index] == res_l].drop(index,axis=1), x[x[index] == res_r].drop(index,axis=1)
            y_l, y_r = y[x[index] == res_l], y[x[index] == res_r]
            left_son = self.rec_build_tree(x_l,y_l,depth+1,res_l)
            right_son = self.rec_build_tree(x_r,y_r,depth+1,res_r)
            sub_entropy = left_son.sub_entropy+right_son.sub_entropy
            nb_leaves = left_son.nb_leaves + right_son.nb_leaves
            return Node(e,p,left_son,right_son,index,label,sub_entropy,nb_leaves)
        else :
            # Else the algo stop and we get a leaf (cf pure with respect to other features)
            return Node(e,p,None,None,"Leaf",label,e,1)


    def CCP(self):
        """ Naive algo that computes alphas and subtrees associated using Cost Complexity Pruning Algo """
        alphas, subtrees = [],[]
        self.CCP_rec(self.root)

    def CCP_rec(self,node,alphas,subtrees):
        if node.nb_leaves == 1:
            # leaf
            pass

    def display(self):
        dot = graphviz.Digraph(comment="Classifier")
        self.display_rec(dot,self.root,0,0)
        path = os.getcwd() + '/Plots/graphviz.txt'
        dot.render(path, view=True)

    def display_rec(self,dot,node,nb,nb_parent):
        if node != None:
            # necesseraly two sons
            dot.node(str(nb), node.name + "\nvalue : " + str(node.value) + "\nEntropy : " + str(node.entropy) + "\nSub Entropy : " + str(node.sub_entropy) + "\nNb Leaves : " + str(node.nb_leaves))
            if nb != 0:
                # if not root, edge from its parent
                dot.edge(str(nb_parent),str(nb), label=node.edge_label)
            self.display_rec(dot,node.left_son,2*nb+1,nb)
            self.display_rec(dot,node.right_son,2*nb+2,nb)

def main():
    rd.seed(1) # for reproductibility
    data = pd.DataFrame({'Sex': rd.choices(["M","F"],k=100),
                         'Study': rd.choices(["W","NW"],k=100),
                         'Courses':rd.choices(["O","NO"],k=100),
                         'Result':rd.choices(["P","F"],k=100)})
    # Entropy test
    Y = data.iloc[:,-1]
    print(entropy(Y))

    # Extracting subsets
    X = data.iloc[:,:-1]
    print(X)
    Y = X.drop(X.columns[0],axis=1)
    print(X)
    print(Y)

    # Extract specific values
    print(X[X.columns[0]].unique())
    new_Y = Y[X[X.columns[0]]=="M"]
    print(new_Y)

    # Splitting algo verif
    classifier = ClassifierTree(1,3)
    classifier.build_tree(data)
    classifier.display()

    # Cost Complexity Pruning
    # interval of values for alpha (pruning parameter)

main()