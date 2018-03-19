#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/7 16:05
# @Author  : MiaFeng
# @Site    : 
# @File    : validation-curve.py
# @Software: PyCharm
__author__ = 'MiaFeng'



'''
Determine whether the model is under-fitting or over-fitting by learning curve, the figure about the (#case_size,scores)
'''

from sklearn.learning_curve import learning_curve

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.cross_validation import train_test_split

def loadData():
    df_wine = pd.read_csv('wine.csv')

    # df_wine.columns = [
    #     'Class label', 'Alcohol',
    #     'Malic acid', 'Ash',
    #     'Alcalinity of ash', 'Magnesium',
    #     'Total phenols', 'Flavanoids',
    #     'Nonflavanoid phenols',
    #     'Proanthocyanins',
    #     'Color intensity', 'Hue',
    #     'OD280/OD315 of diluted wines',
    #     'Proline'
    # ]
    print('Class labels', np.unique(df_wine['Class label']))
    print(df_wine.head())


    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    return X_train, X_test, y_train, y_test

def train(X_train,y_train):
    pipe_lr = Pipeline([
        ('scl', StandardScaler()),
        ('clf', LogisticRegression(penalty='l2', random_state=0))])

    param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    from sklearn.learning_curve import validation_curve

    train_scores, test_scores = validation_curve(estimator=pipe_lr,
                                                            X=X_train,
                                                            y=y_train,
                                                            param_name='clf__C',
                                                            param_range=param_range,
                                                            cv=10)

    # shape(train_scores) = (10,10)???

    return param_range, train_scores,test_scores


def plot_validation_curve(param_range, train_scores, test_scores):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.plot(param_range,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
    plt.fill_between(param_range, train_mean+ train_std, train_mean-train_std,alpha=0.15,color='blue')
    plt.plot(param_range,test_mean,color='green',linestyle='--',marker='s',markersize=5, label='validation accuracy')
    plt.fill_between(param_range,test_mean+test_std, test_mean-test_std, alpha=0.15, color='green')

    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.xscale('log')   # rescale the x-axis in logarithm

    plt.legend(loc='lower right')
    plt.ylim([0.8,1.0])
    plt.show()


if __name__=='__main__':
    sns.set()

    X_train, X_test, y_train, y_test = loadData()

    train_sizes, train_scores, test_scores = train(X_train,y_train)

    plot_validation_curve(train_sizes, train_scores, test_scores)
