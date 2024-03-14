import pandas as pd
import math as m
import graphviz
import os
import random as rd
from collections import deque

def value_obs(observation):
    return (observation == "P").sum()/len(observation)
def error(observation,tot):
    r = min(value_obs(observation),1-value_obs(observation))
    return r * len(observation)/tot
class Node:
    def __init__(self, error=0, value=None, left_son=None, right_son=None, name=None, edge_label=None, sub_error=0, nb_leaves=1, ith=None):
        self.error = error
        self.value = value
        self.name = name
        self.left_son = left_son
        self.right_son = right_son
        self.edge_label = edge_label # for graphviz, label on the edge with its parent
        self.sub_error = sub_error
        self.nb_leaves = nb_leaves
        self.ith = ith # index of the node in a tree

    def copy(self):
        if self.nb_leaves == 1:
            return Node(self.e,self.value,None,None,self.name,self.edge_label,self.sub_error,self.nb_leaves)
        else:
            l,r = self.copy(self.left_son), self.copy(self.right_son)
            return Node(self.e,self.value,l,r,self.name,self.edge_label,self.sub_error,self.nb_leaves)

    def copy_until(self,i):
        """ copy all the tree except the subnode with root at index i """
        if self.ith == i:
            return Node(self.error,self.value,None,None,self.name,self.edge_label,self.sub_error,1,self.ith),self.nb_leaves-1,self.sub_error
        if self.left_son is not None:
            left_son,l_leaves,l_sub_e = self.left_son.copy_until(i)
        else:
            left_son,l_leaves,l_sub_e = None,0,0
        if self.right_son is not None:
            right_son,r_leaves,r_sub_e = self.right_son.copy_until(i)
        else:
            right_son,r_leaves,r_sub_e = None,0,0
        m_leaves = l_leaves + r_leaves # removed leaves
        m_e = l_sub_e + r_sub_e
        self.sub_error = self.sub_error - m_e
        self.nb_leaves = self.nb_leaves - m_leaves
        return Node(self.error,self.value,left_son,right_son,self.name,self.edge_label,self.sub_error,self.nb_leaves,self.ith),m_leaves,m_e

    def g(self):
        if self.nb_leaves == 1:
            return 0
        else:
            return (self.error - self.sub_error) / (self.nb_leaves - 1)

    def set_leaf_rec(self,i):
        if self.name != "Leaf":
            if self.ith == i:
                self.name = "Leaf"
            else:
                self.left_son.set_leaf_rec(i)
                self.right_son.set_leaf_rec(i)

class ClassifierTree:
    def __init__(self, root=None, min_obs=2, max_depth=3):
        self.root = root
        self.min_obs = min_obs
        self.max_depth = max_depth

    def build_tree(self, dataset):
        x, y = dataset.iloc[:,:-1], dataset.iloc[:,-1]
        self.root = self.rec_build_tree(x,y,0,ith=0,tot=len(dataset))

    def rec_build_tree(self,x,y,depth,label=None,ith=None,tot=None):
        p = value_obs(y)
        e = error(y,tot)
        # Stopping criterions : Too deep, too few obs or pure/impure
        if depth >= self.max_depth or len(y) <= self.min_obs or p == 1 or p == 0:
            return Node(e,p,None,None,"Leaf",label,e,1,ith)
        # Else binary split to maximize info gain
        max_info = -float("inf")
        index = None
        for feature_index in x.columns:
            if len(x[feature_index].unique()) == 2:
                res_l, res_r = x[feature_index].unique() # 2 values possible to begin with
                y_l = y[x[feature_index] == res_l] # boolean indexing to dispatch the dataset
                y_r = y[x[feature_index] == res_r]
                # Calculate info gain and select the best dispatch
                info_gain = e - error(y_l,tot) - error(y_r,tot)
                if info_gain > max_info:
                    max_info = info_gain
                    index = feature_index
        if index != None:
            res_l, res_r = x[index].unique()
            x_l, x_r = x[x[index] == res_l].drop(index,axis=1), x[x[index] == res_r].drop(index,axis=1)
            y_l, y_r = y[x[index] == res_l], y[x[index] == res_r]
            left_son = self.rec_build_tree(x_l,y_l,depth+1,res_l,ith=2*ith+1,tot=tot)
            right_son = self.rec_build_tree(x_r,y_r,depth+1,res_r,ith=2*ith+2,tot=tot)
            sub_error = left_son.sub_error+right_son.sub_error
            nb_leaves = left_son.nb_leaves + right_son.nb_leaves
            return Node(e,p,left_son,right_son,index,label,sub_error,nb_leaves,ith)
        else :
            # Else the algo stop and we get a leaf (cf pure with respect to other features)
            return Node(e,p,None,None,"Leaf",label,e,1,ith)

    def set_leaf(self,ith):
        """set the ith node from root to leaf"""
        self.root.set_leaf_rec(ith)

    def CCP(self):
        """ Naive algo that computes alphas and subtrees associated using Cost Complexity Pruning Algo """
        alphas, subtrees = [],[]
        while self.root.nb_leaves != 1:
            # looking for the next "best subtree", minimizing g(t) = (R(t) - R(Tt))/(|Tt|-1)
            queue,id,min_g,leaves = deque([self.root.left_son,self.root.right_son]),0,self.root.g(),self.root.nb_leaves
            while queue: # while not empty
                node = queue.pop()
                if node.name != "Leaf" and (node.g() < min_g or node.g() == min_g and node.nb_leaves < leaves): # tend to take subtrees as large as possible
                    id = node.ith
                    min_g = node.g()
                    leaves = node.nb_leaves
                if node.left_son is not None:
                    queue.append(node.left_son)
                if node.right_son is not None:
                    queue.append(node.right_son)
            alphas.append(min_g)
            subtrees.append(ClassifierTree(root=self.root.copy_until(id)[0]))
            self.root = self.root.copy_until(id)[0]
            self.set_leaf(id)
            #self.display("test", "test")
        return alphas,subtrees

    def display(self,comment,file_name):
        dot = graphviz.Digraph(comment=comment)
        self.display_rec(dot,self.root,0,0)
        path = os.getcwd() + '/Plots/' + file_name + '.txt'
        dot.render(path, view=True)

    def display_rec(self,dot,node,nb,nb_parent):
        if node != None:
            # necesseraly two sons
            dot.node(str(nb), node.name + "\n ith : " + str(node.ith) + "\nvalue : " + str(node.value) + "\nerror : " + str(node.error) + "\nSub error : " + str(node.sub_error) + "\nNb Leaves : " + str(node.nb_leaves))
            if nb != 0:
                # if not root, edge from its parent
                dot.edge(str(nb_parent),str(nb), label=node.edge_label)
            self.display_rec(dot,node.left_son,2*nb+1,nb)
            self.display_rec(dot,node.right_son,2*nb+2,nb)

def main():
    rd.seed(1) # for reproductibility
    data = pd.DataFrame({'Sex': rd.choices(["M","F"],k=1000),
                         'Study': rd.choices(["W","NW"],k=1000),
                         'Courses':rd.choices(["O","NO"],k=1000),
                         'Result':rd.choices(["P","F"],k=1000)})

    data_test = pd.DataFrame({'Sex': ["M","M","F","F","M","F"],
                         'Study': ["W","NW","W","W","NW","NW"],
                         'Courses': ["O","NO","O","NO","NO","NO"],
                         'Result': ["P","F","P","P","P","F"]})
    #data = data_test

    # error test
    #Y = data.iloc[:,-1]
    #print(error(Y,len(Y)))

    # Extracting subsets
    #X = data.iloc[:,:-1]
    #print(X)
    #Y = X.drop(X.columns[0],axis=1)
    #print(X)
    #print(Y)

    # Extract specific values
    #print(X[X.columns[0]].unique())
    #new_Y = Y[X[X.columns[0]]=="M"]
    #print(new_Y)

    # Splitting algo verif
    classifier = ClassifierTree(1,3)
    classifier.build_tree(data)
    classifier.display("Classifier","BasicClassifier")

    # Cost Complexity Pruning
    # interval of values for alpha (pruning parameter)
    #new_root,removed_leaves,s_e = classifier.root.copy_until(3)
    #new_class = ClassifierTree()
    #new_class.root = new_root
    #new_class.display("SubTree3","SubTree3")

    # Test set_leaf
    #classifier.set_leaf(0)
    #classifier.display("Classifier","BasicClassifier")

    # Test functioning Cost Complexity Pruning Algo
    alphas,subtrees = classifier.CCP()
    for i in range(len(alphas)):
        print("alpha = ",alphas[i])
        print("\n")
        #subtrees[i].display(f"subtree {i}",f"subtree {i}")
main()