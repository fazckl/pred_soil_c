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
import numpy as np
import tensorflow as tf
import csv
import pandas as pd
import random as rn


###############################################################################################################



augment = False
n_sample_gen  = 2

aug_ranges = {0: [0.05],        #'C'         
			  1: [100.],        #'max'       
			  2: [100.],        #'median'    
			  3: [100.],        #'min'       
			  4: [15.],         #'sd_stdDev' 
			  5: [8.],          #'aspect'    
			  6: [100.],        #'elevation' 
			  7: [10.],         #'hillshade' 
			  8: [0.5],         #'slope'     
			  9: [80.],         #'B1'        
			  10: [80.],        #'B11'       
			  11: [80.],        #'B12'       
			  12: [80.],        #'B2'        
			  13: [80.],        #'B3'        
			  14: [80.],        #'B4'        
			  15: [80.],        #'B5'        
			  16: [80.],        #'B6'        
			  17: [80.],        #'B7'        
			  18: [80.],        #'B8'        
			  19: [80.],        #'B9'        
			  20: [80.]         #'B10'       
			 }

			 
############### AUGMENTATION GERADE AUS	  -      SOLLTE ABER FUNKTIONIEREN


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
		
    combined_data = np.concatenate((original_data, generated_data), axis=0)
	
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
	else:
		samples = sample_datafrane
	
	scaled_samples = scaler.fit_transform(samples)
	
	input_data = scaled_samples[:,1:].copy()
	labels = scaled_samples[:,:1].copy()
	
	return input_data, labels


#####################################	
	
    
def build_model():
    model = Sequential([Dense(35, input_dim=20, kernel_initializer='normal', activation='relu'),
                        Dense(250, kernel_initializer='normal', activation='relu'),
                        #Dense(350, kernel_initializer='normal', activation='relu'),
                        #Dense(500, kernel_initializer='normal', activation='relu'),
                        #Dense(350, kernel_initializer='normal', activation='relu'),
                        Dense(600, kernel_initializer='normal', activation='relu'),
                        Dense(1, kernel_initializer='normal')
                       ])
    
    model.compile(loss="mean_squared_error", optimizer='Adam')
    return model


def train(train_data, labels):
    seed=5
    np.random.seed(seed)
    
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=build_model, epochs=100, batch_size=5, verbose=1)))
    
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=8, random_state=seed)
    
    results = cross_val_score(pipeline, train_data, labels, cv=kfold)
    print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    
def predict(data):
	pass

	
	
###########################################################################################################	
	
	
	
input_data, labels = prep_data(r'/home/fk/Desktop/Data2.csv')

train(input_data, labels)
