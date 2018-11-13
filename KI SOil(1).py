#!/usr/bin/env python
# coding: utf-8

# In[18]:


import keras
from keras import losses
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# In[3]:


import tensorflow as tf


# In[4]:


import csv


# In[5]:


import pandas as pd


# In[6]:


ReadCsv = pd.read_csv (r'D:\Documents\Unikrahm geordnet\R\KI\Data2.csv')


# In[51]:


df = pd.DataFrame(ReadCsv)
#df = df.iloc[1:,0:]


<<<<<<< HEAD
# In[54]:
=======
# In[109]:
>>>>>>> cb92a8151527364b15667325a84a2351ffafef25


#scaleable df
#scdf = df[['max','median','min','sd_stdDev','aspect','elevation','hillshade','slope','B9','B10']]
<<<<<<< HEAD
scdf = df[['max','median','min','sd_stdDev','aspect','elevation']]
#Wholescale needed on this DF
#nWscdf = df[['B1','B2','B3','B4','B5','B6','B7','B8','B11','B12']]
nWscdf = df[['B1','B2','B3','B4','B5','B6','B7','B8','B11','B12']]


# In[55]:
=======
scdf = np.array(df[['max','median','min','sd_stdDev','aspect','elevation']])
#Wholescale needed on this DF
#nWscdf = df[['B1','B2','B3','B4','B5','B6','B7','B8','B11','B12']]
nWscdf = df[['B2','B3','B4','B5','B6','B7','B8']]


# In[110]:
>>>>>>> cb92a8151527364b15667325a84a2351ffafef25


max=nWscdf.select_dtypes(include=[np.number]).values.max()
min=nWscdf.select_dtypes(include=[np.number]).values.min()
range=max-min
range


<<<<<<< HEAD
# In[56]:
=======
# In[111]:
>>>>>>> cb92a8151527364b15667325a84a2351ffafef25


#MinMaxscaling

#Wholescaled DF
<<<<<<< HEAD
Wscdf=nWscdf.sub(min).divide(range)

=======
#Wscdf=nWscdf.sub(min).divide(range)
Wscdf=scaler.fit_transform(Wscdf)
>>>>>>> cb92a8151527364b15667325a84a2351ffafef25
#Scaling of Independend DF
rnames=list(scdf.index)
cnames=list(scdf)
scaler = MinMaxScaler()
<<<<<<< HEAD
scaler.fit(scdf)
scaled=scaler.transform(scdf)



# In[57]:
=======

scaled=scaler.fit_transform(scdf)



# In[112]:
>>>>>>> cb92a8151527364b15667325a84a2351ffafef25


rnames=scdf.index
cnames=list(scdf)
Wscdf=pd.DataFrame(Wscdf)
scaled=pd.DataFrame(scaled, index=rnames,columns=cnames)
scaled=pd.concat([scaled, Wscdf], axis=1,ignore_index=True)
scaled


<<<<<<< HEAD
# In[58]:


skal_vectoren = np.array(scaled.iloc[0:,0:])
C_vectoren = np.array(scaler.fit_transform(df[['C']]))


# In[59]:
=======
# In[113]:


skal_vectoren = scaled#np.array(scaled.iloc[0:,0:])
C_vectoren = np.array(scaler.fit_transform(df[['C']]))


# In[114]:
>>>>>>> cb92a8151527364b15667325a84a2351ffafef25


C_vectoren


<<<<<<< HEAD
# In[65]:
=======
# In[115]:
>>>>>>> cb92a8151527364b15667325a84a2351ffafef25


model = Sequential([
					Dense(20, input_shape=(skal_vectoren.shape[1],), activation='relu'),
<<<<<<< HEAD
					Dense(100, activation='relu'),
					Dense(30, activation='relu'),
					Dense(1, activation='relu')
				   ])

				   
model.compile(Adam(lr=.0001), loss='mean_squared_error', metrics=['accuracy'])
=======
					Dense(1000, activation='relu'),
					Dense(600, activation='relu'),
					Dense(1, activation='sigmoid')
				   ])

				   
model.compile(Adam(lr=.1), loss='mean_squared_error', metrics=['accuracy'])
>>>>>>> cb92a8151527364b15667325a84a2351ffafef25

model.fit(skal_vectoren,C_vectoren, epochs=10, batch_size=6, validation_split=10,)


# In[ ]:





# In[ ]:





# In[ ]:




