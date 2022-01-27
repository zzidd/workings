#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as ppy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier


# In[4]:


url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
names = ['sepal-length','sepal-width','petal-length','petal-width','class']
dataset = read_csv(url,names = names)
print(dataset.shape)


# In[6]:


dataset.head(20)


# In[13]:


dataset.groupby('class').size()


array = dataset.values
X = array[:,0:4]
Y = array[:,4]
X_train,X_validation,Y_train,Y_validation = train_test_split(X,Y,test_size = 0.2,random_state = 1)


# In[34]:


models = []
models.append(('LR',LogisticRegression(solver = 'liblinear',multi_class = 'ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))


# In[38]:


results = []
names = []
for name,model in models:
    kfold = StratifiedKFold(n_splits = 10)
    cv_results =  cross_val_score(model,X_train,Y_train,cv = kfold,scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name,cv_results.mean(),cv_results.std()))


# In[40]:


ppy.boxplot(results,labels = names)
ppy.title('Algortihm Comparision')


# In[42]:


model = SVC(gamma='auto')
model.fit(X_train,Y_train)
predictions = model.predict(X_validation)


# In[43]:


print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))

