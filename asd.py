#!/usr/bin/env python
# coding: utf-8

# In[9]:


from __future__ import unicode_literals
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.callbacks import History
hist = History()
from keras import losses
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.optimizers import Adam
from keras.metrics import mean_squared_error as mean_squared_error_keras
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import keras
import numpy as np
import tensorflow as tf
import csv
import pandas as pd
import random as rn
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.utils import np_utils, generic_utils
from random import shuffle
import pickle
import math
from sklearn.metrics import mean_squared_error as mean_squared_error_sklearn
from numpy import shape as shape


# In[10]:



augment = False
n_sample_gen  = 1
usepickle=False

aug_ranges = {0: [0.005],        #'C'         
              1: [10.],        #'max'       
              2: [10.],        #'median'    
              3: [10.],        #'min'       
              4: [1.5],         #'sd_stdDev' 
              5: [.8],          #'aspect'    
              6: [10.],        #'elevation' 
              7: [1.],         #'hillshade' 
              8: [0.05],         #'slope'     
              9: [8.],         #'B1'        
              10: [8.],        #'B11'       
              11: [8.],        #'B12'       
              12: [8.],        #'B2'        
              13: [8.],        #'B3'        
              14: [8.],        #'B4'        
              15: [8.],        #'B5'        
              16: [8.],        #'B6'        
              17: [8.],        #'B7'        
              18: [8.],        #'B8'        
              19: [8.],        #'B9'        
              20: [8.]         #'B10'       
             }

 


# In[11]:


def aug_data(data_frame):
    
    
    generated_data = []

    for v in original_data:
        for n in range(n_sample_gen):
            generated_sample = []
            for i, value in enumerate(v):
                if i == 0:
                    generated_sample.append(value)
                else:
                    generated_sample.append(value+rn.uniform(-aug_ranges[i][0], -aug_ranges[i][0]))
            generated_data.append(generated_sample)

    combined_data = generated_data
    #combined_data = np.concatenate((original_data, generated_data), axis=0)
    

    return combined_data, vali


# In[12]:


def prep_data(path1, path2):

   
        
        Dataunsortet = pd.read_csv(path1)
        Cunsortet = pd.read_csv(path2)

        CSortet = Cunsortet.sort_values(by=['OBJECTID'])
        Datasortet = Dataunsortet.sort_values(by=['OBJECTID'])
        Concatet = pd.concat([Cunsortet, Dataunsortet], axis=1)
        #print(Concatet,'Concatet')
        IN=list(Concatet.columns.difference(['index','system:index','OBJECTID','.geo']))
        Trimmed = Concatet.loc[:, IN]
        Trimmed = Trimmed.loc[:,['LOI', 'B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
       'B8A', 'B8_1', 'B9', 'LSTmax', 'LSTmedian', 'LSTmin', 'SIndmax',
       'SIndmedian', 'SIndmin', 'aspect', 'elevation', 'hillshade', 'ndvimax',
       'ndvimedian', 'ndvimin', 'sd_stdDev', 'sd_stdDev_1', 'sd_stdDev_2',
       'slope']]
        #print(type(Trimmed),'Trimmed')
        #print(Trimmed.columns,'Trimmed')
        scaler = StandardScaler()
        scaler_v = StandardScaler()
       
        y = Trimmed.iloc[:,0].reshape(1, -1)
        X = Trimmed.iloc[:,1:]
        print(X.columns,'X')
        
                
        if augment is True:
            x = aug_data(original_data)

        else:
           
            y = scaler_v.fit_transform(y)
            X = scaler.fit_transform(X)
            #Scaled=scaler.fit_transform(np.array(Trimmed))
        
       
       # X=Scaled[:,1:]
        #y=Scaled[:,0]
        y = y.reshape(-1, 1)
        
        trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.15, random_state=42)
        
    

        val= tuple([testX, testY])

        
 
        return trainX, trainY, val, testX, testY, scaler_v


# In[13]:


def build_model():
    model = Sequential([Dense(100, input_dim=firstdim, kernel_initializer='normal', activation='relu'),
                        Dense(500, kernel_initializer='normal', activation='relu'),
                        Dropout(drop1, noise_shape=None, seed=54),
                        Dense(2000, kernel_initializer='normal', activation='relu'),
                       # Dense(2000, kernel_initializer='normal', activation='relu'),
                        Dropout(drop2, noise_shape=None, seed=54),
                        #Dense(200, kernel_initializer='normal', activation='relu'),
                        #Dense(300, kernel_initializer='normal', activation='relu'),
                        #Dense(500, kernel_initializer='normal', activation='relu'),
                        Dense(2000, kernel_initializer='normal', activation='relu'),
                        Dropout(drop3, noise_shape=None, seed=54),
                        Dense(600, kernel_initializer='normal', activation='relu'),
                        Dense(1, kernel_initializer='normal')
                       ])


    adam = keras.optimizers.Adam(lr=learnrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)    
    model.compile(loss="mean_squared_error", optimizer=adam)
    return model


# In[14]:


def train(train, lab, val,nb_epoch,firstdim):
    print(shape(train[0]),'shape')
    model = build_model()
    hist = model.fit(train, lab, epochs=nb_epoch, validation_data=val, verbose=1, batch_size=6)


    #seed=5
    #np.random.seed(seed)
    
    #estimators = []
    #estimators.append(('standardize', StandardScaler()))
    #estimators.append(('mlp', KerasRegressor(build_fn=build_model, epochs=100, batch_size=5, verbose=1)))
    
    #pipeline = Pipeline(estimators)
    #kfold = KFold(n_splits=8, random_state=seed)
    
    #results = cross_val_score(pipeline, train_data, labels, cv=kfold)
    #print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    return hist, model


def predict(data):
    pass



# In[15]:


drop1=0.05
drop2=0.01
drop3=0.005
learnrate=0.000001
nb_epoch=20

k = 1

train_loss_sum = 0
val_loss_sum = 0

for i in range(k):
    input_data, labels, val, testX, testY, scaler_v = prep_data(r'D:\Documents\Unikrahm geordnet\R\KI/Inputs.csv',r'D:\Documents\Unikrahm geordnet\R\KI/AimsC.csv')
    firstdim=(shape(input_data))[1]
    print(type(firstdim),'type')
    hist, model = train(input_data, labels, val,nb_epoch,firstdim)

    #train_loss  =  hist.history['loss']
    #val_loss =  hist.history['val_loss']
    
    
    
    trainX = input_data
    
    testX = testX
    trainY=labels
    
    testY=testY
    #test this model
    trainPredictions = model.predict(trainX)
    testPredictions =model.predict(testX)
    
    predictions = np.append(trainPredictions, testPredictions)#, axis=0)
    
    original_data = np.append(trainY, testY)
    
    
    unscaled_predictions = scaler_v.inverse_transform(predictions)
    #unscale
    unscaled_trainP = unscaled_predictions[:shape(trainPredictions)[0]]
    unscaled_testP  = unscaled_predictions[shape(trainPredictions)[0]:]
    
    
    #testPredictions
    unscaled_orig_data = scaler_v.inverse_transform(original_data)
    
    unscaled_trainY = unscaled_orig_data[:shape(trainY)[0]]
    unscaled_testY = unscaled_orig_data[shape(trainY)[0]:]
   
    
    
    #calculate the RMSE
    
    trainScore = math.sqrt(mean_squared_error_sklearn(unscaled_trainY,unscaled_trainP))
    
    testScore = math.sqrt(mean_squared_error_sklearn(unscaled_testY,unscaled_testP))
    print('Train score: rmse', trainScore)
    print('Test score: rmse', testScore)
    
    train_loss_sum += trainScore
    val_loss_sum += testScore
    
print(train_loss_sum/k)
print(val_loss_sum/k)
    
    
    


# In[13]:


train_loss=hist.history['loss']
val_loss=hist.history['val_loss']

xc=range(nb_epoch)

drop1='d1'+repr(drop1)
drop2='d2'+repr(drop2)
drop3='d3'+repr(drop3)
learnrate='lr'+repr(learnrate)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')#+drop1+drop2+drop3+learnrate)
plt.grid(True)
plt.legend(['train','val'])
print (plt.style.available) # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.show()


# In[14]:


trainX = input_data
testX = testX
trainY=labels
testY=testY
#test this model
trainPredictions = model.predict(trainX)
testPredictions =model.predict(testX)


predictions = np.append(trainPredictions, testPredictions)#, axis=0)
unscaled_predictions = scaler_v.inverse_transform(predictions)
#unscale
unscaled_trainP = unscaled_predictions[:shape(trainPredictions)[0]]
unscaled_testP  = unscaled_predictions[shape(trainPredictions)[0]:]
print(unscaled_testP)

originals = np.append(trainY,testY)
unscaledorigs=scaler_v.inverse_transform(originals)
unscaled_trainO = unscaledorigs[:shape(trainY)[0]]
unscaled_testO  = unscaledorigs[shape(trainY)[0]:]
#testPredictions
#trainY = scaler_v.inverse_transform(trainY)
#testY=scaler_v.inverse_transform(testY)


#calculate the RMSE

trainScore = math.sqrt(mean_squared_error_sklearn(unscaled_trainO,unscaled_trainP))

testScore = math.sqrt(mean_squared_error_sklearn(unscaled_testO,unscaled_testP))
print('Train score: rmse', trainScore)
print('Test score: rmse', testScore)


# In[15]:


plt.plot(unscaled_testP.flatten(),unscaled_testO,'kx')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




