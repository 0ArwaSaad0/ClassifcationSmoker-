# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 21:09:17 2022

@author: Arwa
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt




#read Data
data = pd.read_csv('insurance.csv')

#understan data
data.head()
data.tail()
data.shape
data.columns

#check missing data
data.isnull().sum()

#descriptive analysis
data.describe()

#corellation

Correlation = data.corr()
print(Correlation)

#heatmap
sns.heatmap(Correlation,xticklabels=Correlation.columns,yticklabels=Correlation.columns,annot=True)



#outlayer of charges
sns.catplot(x='charges',kind='box',color='steelblue',data=data)
No_Outliers=data[data.charges<=20000]
sns.catplot(x='charges',kind='box',color='steelblue',data=No_Outliers)

#out layer of children
sns.catplot(x='children',kind='box',data= data,color='seagreen')

#out layer of bmi
sns.catplot(x='bmi',kind='box',data=data)

No_Outliers=(data[data['bmi']>47])
sns.catplot(x='bmi',kind='box',data=No_Outliers)



data['region'].unique()
#labeled
data.smoker.replace(('yes','no'),(1,0), inplace=True)
data.sex.replace(('female','male'),(1,0), inplace=True)
data.region.replace(('southwest', 'southeast', 'northwest', 'northeast'),(1,2,3,4), inplace=True)

data.head()


"""Builng Model"""

# Splitting the data

X = data[['age','sex','region','children','charges','bmi']]
y = data[['smoker']]


X.head()
y.head()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

"""KNN"""
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_neighbors= 15)
knn_clf.fit(X_train,y_train)
print(knn_clf.score(X_test,y_test))


y_pred_knn = knn_clf.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
Confusion_m = confusion_matrix(y_test,y_pred_knn)
print(Confusion_m)
print("Accuracy KNeighbors:",accuracy_score(y_test,y_pred_knn))
print(classification_report(y_test, y_pred_knn))
"""SVM"""

from sklearn.svm import SVC

sv_clf = SVC(probability=True,kernel='sigmoid')
sv_clf.fit(X_train,y_train)
print(sv_clf.score(X_test,y_test))

y_pred_sv = sv_clf.predict(X_test)
Confusion_m = confusion_matrix(y_test,y_pred_sv)
Confusion_m

print("Accuracy Of SVM:",accuracy_score(y_test,y_pred_sv)*100)
print(classification_report(y_test, y_pred_sv))

