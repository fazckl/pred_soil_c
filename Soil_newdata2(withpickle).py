from __future__ import unicode_literals
from keras.callbacks import History
hist = History()
from keras import losses
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.optimizers import Adam
from keras.metrics import mean_squared_error
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
from keras.utils import np_utils, generic_utils
from random import shuffle
import pickle



###############################################################################################################



augment = False
n_sample_gen  = 1
usepickle=True

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

			 
############### 


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

##################################### HABE DOCH DIE NORMALISIERUNG RAUSGENOMMEN, WEIL ICH MICH DA VERTAN HAB   -    STANDARDISIERUNG REICHT VIELLEICHT WIE DU MEINTEST  
##################################### DAS IST JA DANN ALLES ETWA IM UMFELD VON KLEINEN WERTEN    -    DIE FRAGE IST NUR OB WEIT ENTFERNTE OUTLIER PROBLEME MACHEN KÖNNTEN
##################################### DESWEGEN IST HIER AUCH DIE EINTEILUNG IN GRUPPEN NICHT NÖTIG

#####################################  HIERIN SIND 2 PATHS; EINMAL DEN c WERT UND EINMAL DIE DATEN: SIE WERDEN SO SORTIERT DAS DER C WERT AM ANFANG STEHT; DAS HABE ICH
##################################### GEMACHT UM DEN SPÄTER GEGEN ANDERE PARAMETER ZU TAUSCHEN UND VLLT BESSERE ERGEBNISSE FÜR ANDERE ZIELWERTE ZU ERHALTEN

def prep_data(path1, path2):

    if usepickle == False:
        print('Fresh Cucumber')
        Dataunsortet = pd.read_csv(path1)
        Cunsortet = pd.read_csv(path2)
        print(Cunsortet)

        CSortet = Cunsortet.sort_values(by=['OBJECTID'])
        Datasortet = Dataunsortet.sort_values(by=['OBJECTID'])
        Concatet = pd.concat([Cunsortet, Dataunsortet], axis=1)
        IN=list(Concatet.columns.difference(['system:index','OBJECTID','.geo']))
        Trimmed = Concatet.loc[:, IN]
        ElevSorted = Trimmed.sort_values(by=['elevation'])
        length = ElevSorted.shape
        i = 0
        vali=pd.DataFrame()
        test=pd.DataFrame()

        
        while i<length[0]:
            temp=ElevSorted.iloc[i:i+5,:]
            temp=temp.reset_index()
            vali=vali.append(temp.iloc[0,:])
            test=test.append(temp.iloc[1:4,:])
            i+=5
        
        pickle.dump( test, open( "test1.5.p", "wb" ) )
        pickle.dump( vali, open( "vali1.5.p", "wb" ) )
        
       # test.to_csv(r'D:\Documents\Unikrahm geordnet\R\KI\Figures/Inputs1_3.csv')
       # vali.to_csv(r'D:\Documents\Unikrahm geordnet\R\KI\Figures/Vali1_3.csv')
        original_data = np.array(test)
        #original_data = np.random.shuffle(original_data.flat) 
        vali = np.array(vali)
       
   
	
        scaler = StandardScaler()
	
        if augment is True:
            samples = aug_data(original_data)
		
        else:
            samples = original_data
		
        val = scaler.fit_transform(vali)
        scaled_samples = scaler.fit_transform(samples)         
	
        input_data = scaled_samples[:,1:].copy()
        labels = scaled_samples[:,:1].copy()

        orig_samples= val[:,1:].copy()
        orig_labels = val[:,:1].copy()
        val= tuple([orig_samples, orig_labels])
	
        
    else:
        print("I´m Pickle Rick !")
        test = pickle.load( open( "test1.5.p", "rb" ) )
        vali = pickle.load( open( "vali1.5.p", "rb" ) )
        original_data = np.array(test)
        #original_data = np.random.shuffle(original_data.flat) 
        vali = np.array(vali)
       
   
	
        scaler = StandardScaler()
	
        if augment is True:
            samples = aug_data(original_data)
		
        else:
            samples = original_data
		
        val = scaler.fit_transform(vali)
        scaled_samples = scaler.fit_transform(samples)         
	
        input_data = scaled_samples[:,1:].copy()
        labels = scaled_samples[:,:1].copy()

        orig_samples= val[:,1:].copy()
        orig_labels = val[:,:1].copy()
        val= tuple([orig_samples, orig_labels])

    return input_data, labels, val

#####################################	
	
    
def build_model():
    model = Sequential([Dense(100, input_dim=31, kernel_initializer='normal', activation='relu'),
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


###################################  HAUPTPROBLEM WAR, DASS LOSS-FUNCTION UND METRICS AUF EIN
###################################  KATEGORISIERUNGSPROBLEM AUSGELEGT WAREN WIR ABER EIN REGRESSIONSPROBLEM HABEN


def train(train, lab, val,nb_epoch):
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

    return hist
def predict(data):
	pass

	
	
###########################################################################################################	
	
drop1=0.05
drop2=0.01
drop3=0.005
learnrate=0.0000001
	
input_data, labels, val = prep_data(r'D:\Documents\Unikrahm geordnet\R\KI/Inputs.csv',r'D:\Documents\Unikrahm geordnet\R\KI/AimsC.csv')
nb_epoch=2
print(val)
hist = train(input_data, labels, val,nb_epoch)

####################Plot##########################
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
#print (plt.style.available) # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.show()


trainX = input_data
testX = pd.DataFrame(val).iloc[1:,:]
trainY=labels
testY=pd.DataFrame(val).iloc[:0,:]
#test this model
trainPredictions = model.predict(trainX)
testPredctions =model.predict(testX)
#unscale
trainPredictions=scaler.inverse_transform(trainPredictions)
testPredictions=scaler.inverse_transform(testPredictions)
trainY = scaler.inverse_transform(trainY)
testY=scaler.inverse_transform(testY)


#calculate the RMSE
trainScore = math.sqrt(mean_squared_error(trainY[0],trainPredictions[:,0]))
testScore = math.sqrt(mean_squared_error(testY[0],testPredictions[:,0]))
print('Train score: rmse', trainScore)
type(trainScore)
print('Test score: rmse', testScore)

