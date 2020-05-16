#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle


# In[2]:


data = pd.read_csv("E:\\Expenses_prediction.csv")


# In[3]:


data.shape


# In[4]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
data_scaled = scaler.fit_transform(data)
df = pd.DataFrame.from_records(data_scaled)


# In[5]:


X = df.iloc[:,:2]


# In[6]:


Y= df.iloc[:,-1]


# In[7]:


from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X, Y)


# In[8]:


from sklearn.metrics import r2_score
Y_predict = lr_model.predict(X)
r2 = r2_score(Y, Y_predict)
print('R2 score is {}'.format(r2))


# In[10]:


# Saving model to disk
pickle.dump(lr_model, open('E:\\lr_model.pkl','wb'))


# In[ ]:




