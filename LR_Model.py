#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
import pickle


# In[17]:


data = pd.read_csv("E:\\Build and Deploy ML\\Deploy in Flask\\Consumer1.csv")


# In[18]:


data.shape


# In[19]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
data_scaled = scaler.fit_transform(data)
df = pd.DataFrame.from_records(data_scaled)


# In[20]:


X = df.iloc[:,:2]


# In[21]:


Y= df.iloc[:,-1]


# In[22]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score

lr_model = LinearRegression()
lr_model.fit(X, Y)


# In[11]:


#Y_predict = lr_model.predict(X)


# In[12]:


#rmse = (np.sqrt(mean_squared_error(Y, Y_predict)))


# In[13]:


#r2 = r2_score(Y, Y_predict)


# In[14]:


#print('RMSE is {}'.format(rmse))
#print('R2 score is {}'.format(r2))
#print("\n")


# In[23]:


# Saving model to disk
pickle.dump(lr_model, open('E:\\Build and Deploy ML\\Deploy in Flask\\model.pkl','wb'))


# In[ ]:




