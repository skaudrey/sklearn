#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/6 18:58
# @Author  : MiaFeng
# @Site    : 
# @File    : preprocess.py
# @Software: PyCharm
__author__ = 'MiaFeng'

import pandas as pd
from io import StringIO
import numpy as np

def demo_NaN():
    csv_data = '''A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    0.0,11.0,12.0,
    '''
    df = pd.read_csv(StringIO(csv_data))
    print(df)

    sum = df.isnull().sum() # the number of missing data in each column
    print(sum)

    drop_row_any_null = df.dropna()
    print(drop_row_any_null)

    # drop column where has at least one NaN
    print(df.dropna(axis=1))

    # only drop rows where all column are NaN
    print(df.dropna(how='all'))

    # drop rows that have not at least 4 non-NaN values
    print(df.dropna(thresh=4))

    # drop rows where NaN appear in specific columns (here: 'C')
    print(df.dropna(subset=['C']))

    # fill missing data by the mean this feature
    from sklearn.preprocessing import Imputer
    imr = Imputer(missing_values='NaN',strategy='mean',axis=0)
    imr = imr.fit(df)
    print(imr.transform(df.values))

def demo_label():
    df = pd.DataFrame([
        ['green','M',10.1,'class1'],
        ['red','L',13.5,'class2'],
        ['blue','XL',15.3,'class1']])
    df.columns = ['color','size','price','classlabel']
    print(df)

    # mapping the ordered feature
    size_mapping = {
        'XL':3,
        'L':2,
        'M':1
    }
    df['size'] = df['size'].map(size_mapping)
    print(df)

    # mapping the unordered data by invert mapping dict
    class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
    df['classlabel'] = df['classlabel'].map(class_mapping)
    inv_class_mapping = {v:k for k,v in class_mapping.items()}
    df['classlabel'] = df['classlabel'].map(inv_class_mapping)
    print(df)

    # mapping the unordered feature by LabelEncoder
    from sklearn.preprocessing import LabelEncoder
    class_le = LabelEncoder()
    df['classlabel'] = class_le.fit_transform(df['classlabel'].values)
    print(df)

    df['classlabel'] = class_le.inverse_transform(df['classlabel'].values)
    # print(class_le.inverse_transform(df['classlabel'].values))
    print(df)

    # one-hot
    from sklearn.preprocessing import OneHotEncoder
    one = OneHotEncoder(categorical_features=[0])   # the feature columns we want to transform
    X = df[['color','size','price']].values
    X[:,0] = LabelEncoder().fit_transform(X[:,0])
    print(one.fit_transform(X).toarray())

    # one-hot by dummies
    print(pd.get_dummies(df[['color','size','price']]))

def feature_scaling():
    X = np.random.random((20,2))
    print(X)
    X_train = X[0:13,:]
    X_test = X[13:,:]
    from sklearn.preprocessing import MinMaxScaler
    # normalization

    mms = MinMaxScaler()
    x_train_norm = mms.fit_transform(X_train)
    print(x_train_norm)
    print('\n')
    x_test_norm = mms.transform(X_test)
    print(x_test_norm)
    print('\n')

    # standardScaler
    from sklearn.preprocessing import StandardScaler
    stdsc = StandardScaler()
    x_train_std = stdsc.fit_transform(X_train)
    print('\n')
    print(x_train_std)
    x_test_std = stdsc.transform(X_test)
    print('\n')
    print(x_test_std)
    print('\n')


def chooose_feature_rf():
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)

    df_wine.columns = [
        'Class label','Alcohol',
        'Malic acid','Ash',
        'Alcalinity of ash','Magnesium',
        'Total phenols','Flavanoids',
        'Nonflavanoid phenols',
        'Proanthocyanins',
        'Color intensity','Hue',
        'OD280/OD315 of diluted wines',
        'Proline'
    ]
    print('Class labels',np.unique(df_wine['Class label']))
    print(df_wine.head())

    df_wine.to_csv('wine.csv')


    from sklearn.cross_validation import train_test_split
    X,y = df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

    from sklearn.ensemble import RandomForestClassifier
    feat_labels = df_wine.columns[1:]
    print(feat_labels)
    forest = RandomForestClassifier(n_estimators=10000,random_state=0,n_jobs=-1)
    forest.fit(X_train,y_train)

    importances = forest.feature_importances_
    temp =  np.argsort(importances)
    indices = temp[::-1]    #invert

    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f"%(f+1, 30, feat_labels[f],importances[indices[f]]))


    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    plt.figure()
    plt.title('Feature Importances')
    plt.bar(range(X_train.shape[1]),importances[indices],color='lightblue',align='center')
    plt.xticks(range(X_train.shape[1]),feat_labels,rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.show()



if __name__=='__main__':
    # demo_NaN()
    # demo_label()
    # feature_scaling()
    chooose_feature_rf()