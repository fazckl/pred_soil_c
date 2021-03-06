from keras.callbacks import History
hist = History()
from keras import losses
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
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


###############################################################################################################



augment = True
n_sample_gen  = 1

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
    original_data = np.array(data_frame)
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
    
	
    return combined_data

##################################### HABE DOCH DIE NORMALISIERUNG RAUSGENOMMEN, WEIL ICH MICH DA VERTAN HAB   -    STANDARDISIERUNG REICHT VIELLEICHT WIE DU MEINTEST  
##################################### DAS IST JA DANN ALLES ETWA IM UMFELD VON KLEINEN WERTEN    -    DIE FRAGE IST NUR OB WEIT ENTFERNTE OUTLIER PROBLEME MACHEN KÖNNTEN
##################################### DESWEGEN IST HIER AUCH DIE EINTEILUNG IN GRUPPEN NICHT NÖTIG

def prep_data(path):
	ReadCsv = pd.read_csv(path)
	
	sample_datafrane = pd.DataFrame(ReadCsv, columns=['C', 'max', 'median', 'min', 'sd_stdDev', 'aspect', 'elevation', 'hillshade', 'slope', 'B1', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10'])
	
	scaler = StandardScaler()
	
	if augment is True:
		samples = aug_data(sample_datafrane)
		val = sample_datafrane
	else:
		samples = sample_datafrane
	val = scaler.fit_transform(val)
	scaled_samples = scaler.fit_transform(samples)         
	
	input_data = scaled_samples[:,1:].copy()
	labels = scaled_samples[:,:1].copy()

	orig_samples= val[:,1:].copy()
	orig_labels = val[:,:1].copy()
	val= tuple([orig_samples, orig_labels])
	
	return input_data, labels, val


#####################################	
	
    
def build_model():
    model = Sequential([Dense(100, input_dim=20, kernel_initializer='normal', activation='relu'),
                        Dense(200, kernel_initializer='normal', activation='relu'),
                        Dense(300, kernel_initializer='normal', activation='relu'),
                        #Dense(500, kernel_initializer='normal', activation='relu'),
                        #Dense(2000, kernel_initializer='normal', activation='relu'),
                        Dense(600, kernel_initializer='normal', activation='relu'),
                        Dense(1, kernel_initializer='normal')
                       ])
    
    model.compile(loss="mean_squared_error", optimizer='Adam')
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
	
	
	
input_data, labels, val = prep_data(r'D:\Documents\Unikrahm geordnet\R\KI/Data2.csv')
nb_epoch=200
hist = train(input_data, labels, val,nb_epoch)

####################Plot##########################
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
xc=range(nb_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print (plt.style.available) # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.show()



