#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/6 18:43
# @Author  : MiaFeng
# @Site    : 
# @File    : rf.py
# @Software: PyCharm
__author__ = 'MiaFeng'


from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from util.plot_util import plot_decision_regions
from util.path_util import currentPath
import numpy as np
# import matplotlib.pyplot as plt
import seaborn as sns

def loadData():
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    return X, y


def rf():
    X, y = loadData()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    tree = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1,n_jobs=2)
    tree.fit(X_train, y_train)
    x_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    basepath = currentPath() + '/'

    plot_decision_regions(x_combined, y_combined, classifier=tree, basepath=basepath, fileName='randomForest.png')

    # export decision image
    from sklearn.tree import export_graphviz
    export_graphviz(tree, out_file='rf.dot', feature_names=['petal length', 'petal width'])

    # you can change it to .png file by
    # >>> dot -Tpng tree.dot -o tree.png in your terminal
    # Attention: I have installed dot and set the environment of my system, thus I can use dot in my terminal directly

if __name__=='__main__':
    sns.set()
    rf()