#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# In[2]:


import tensorflow as tf


# In[3]:


import csv


# In[4]:


import pandas as pd


# In[5]:


ReadCsv = pd.read_csv (r'D:\Documents\Unikrahm geordnet\R\KI\Data2.csv')


# In[6]:


df = pd.DataFrame(ReadCsv)
df


# In[7]:


#scaleable df
scdf = df[['max','median','min','sd_stdDev','aspect','elevation','hillshade','slope','B9','B10']]
#Wholescale needed on this DF
nWscdf = df[['B1','B2','B3','B4','B5','B6','B7','B8','B11','B12']]


# In[8]:


max=nWscdf.select_dtypes(include=[np.number]).values.max()
min=nWscdf.select_dtypes(include=[np.number]).values.min()
range=max-min
range


# In[9]:


#MinMaxscaling

#Wholescaled DF
Wscdf=nWscdf.sub(min).divide(range)

#Scaling of Independend DF
rnames=list(scdf.index)
cnames=list(scdf)
scaler = MinMaxScaler()
scaler.fit(scdf)
scaled=scaler.transform(scdf)


# In[10]:


rnames=scdf.index
cnames=list(scdf)
scaled=pd.DataFrame(scaled, index=rnames,columns=cnames)
print(scaled)


# In[11]:


skal_vectoren = np.array(scaled.iloc[1:,2:])
C_vectoren = np.array(df[['C']])


# In[12]:


C_vectoren


# In[15]:


model = Sequential([
					Dense(21, input_shape=(21,), activation='relu'),
					Dense(100, activation='relu'),
					Dense(30, activation='relu'),
					Dense(1, activation='softmax')
				   ])

				   
model.compile(Adam(lr=.0001, loss='mean_squared_error', metrics['accuracy']))

model.fit(skal_vectoren,C_vectoren, epochs=10, batch_size=6, validation_split=10, verbose=2)
model.fit(skal_vectoren,C_vectoren, epochs=10, batch_size=15, validation_split=10)


# In[ ]:




