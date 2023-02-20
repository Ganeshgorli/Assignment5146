#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle
#from sklearn.linear_model import LinearRegression


# In[2]:


dataset = pd.read_csv('BankNote_Authentication.csv')


# In[3]:


X = dataset.iloc[:, :4]
y = dataset.iloc[:, -1]


# In[6]:


regressor = DecisionTreeClassifier()


# In[7]:


regressor.fit(X, y)


# In[8]:


pickle.dump(regressor, open('BankNote.pkl','wb'))


# In[9]:


model = pickle.load(open('BankNote.pkl','rb'))


# In[10]:


print(model.predict([[3.6216, 8.6661, -2.8073, -0.44699]]))


# In[ ]:

