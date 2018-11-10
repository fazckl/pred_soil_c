from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import mean_squared_error


import numpy as np
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import csv
import pandas as pd

def prep_data(path):
	ReadCsv = pd.read_csv (r'D:\Documents\Unikrahm geordnet\R\KI\Data2.csv')
	
	df = pd.DataFrame(ReadCsv)
	
	scdf = df[['max','median','min','sd_stdDev','aspect','elevation','hillshade','slope','B9','B10']]
	nWscdf = df[['B1','B2','B3','B4','B5','B6','B7','B8','B11','B12']]
	
	max=nWscdf.select_dtypes(include=[np.number]).values.max()
	min=nWscdf.select_dtypes(include=[np.number]).values.min()
	range=max-min
	
	Wscdf = nWscdf.sub(min).divide(range)
	
	rnames=list(scdf.index)
	cnames=list(scdf)
	
	scaler = MinMaxScaler()
	scaler.fit(scdf)
	scaled=scaler.transform(scdf)
	
	rnames=scdf.index
	cnames=list(scdf)
	
	scaled=pd.DataFrame(scaled, index=rnames,columns=cnames)
	print(scaled)
	
	skal_vects = np.array(scaled.iloc[1:,2:])
	C_vects = np.array(df[['C']])
	
	return skal_vects, c_vects
	


def train():
	model = Sequential([
						Dense(21, input_shape=(21,), activation='relu'),
						Dense(100, activation='relu'),
						Dense(30, activation='relu'),
						Dense(1, activation='softmax')
					])
	
					
	model.compile(Adam(lr=.0001), loss='mean_squared_error', metrics=['accuracy']))
	
	model.fit(skal_vects,C_vects, epochs=10, batch_size=6, validation_split=10, verbose=2)
	
def predict(input_vec):
	






