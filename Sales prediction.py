#!/usr/bin/env python
# coding: utf-8

# # Sales Prediction

# #### 1. Importing modules

# In[20]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# #### 2. Importing & analyzing dataset

# In[21]:


sales = pd.read_csv('Advertising.csv')
sales.head()


# In[22]:


# removing unecessary columns
sales=sales.drop(columns = ['Unnamed: 0'])
sales.head()


# In[23]:


# dataset dimensions
sales.shape


# In[24]:


# information of dataset
sales.info()


# In[25]:


# checking for null values
sales.isnull().sum()


# In[26]:


# stats of data
sales.describe()


# #### 3. Graphs & Charts 

# In[27]:


# TV
sales['TV'].hist(color='Purple')


# In[28]:


# TV
sales['Radio'].hist(color='Green')


# In[29]:


# Newspaper
sales['Newspaper'].hist(color='Pink')


# In[30]:


# Pie chart
tv=sum(sales['TV'])
radio=sum(sales['Radio'])
newspaper = sum(sales['Newspaper'])
sale = np.array([tv,radio,newspaper])
lab=['TV','Radio','Newspaper']
myexplode=[0.1,0,0]
col=['Brown','Blue','Grey']
plt.pie(sale, labels= lab,explode= myexplode,shadow=True,colors=col)
plt.show()


# #### 4. Correlation

# In[31]:


sales.corr()


# In[32]:


# mapping correlation
corr=sales.corr()
fig, ax= plt.subplots(figsize=(3,3))
sns.heatmap(corr, annot=True, ax=ax)


# #### 5. Model Training

# In[33]:


X= sales.drop(columns=['Sales'])
Y=sales['Sales']
X_train,X_test, Y_train,Y_test= train_test_split(X,Y,test_size=0.20,random_state=3)


# In[34]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)
print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[35]:


# Regression model
reg = LinearRegression()
reg.fit(X_train,Y_train)
Y_pred= reg.predict(X_train)


# In[36]:


print("Accuracy",reg.score(X_train,Y_train)*100)


# In[37]:


# Regression model
reg = LinearRegression()
reg.fit(X_test,Y_test)
Y_pred= reg.predict(X_test)


# In[38]:


print("Accuracy",reg.score(X_test,Y_test)*100)


# In[ ]:




