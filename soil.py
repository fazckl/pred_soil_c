import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import csv
import pandas as pd

ReadCsv = pd.read_csv (r'D:\Documents\Unikrahm geordnet\R\KI\Data2.csv')

df = pd.DataFrame(ReadCsv)

#scaleable df
scdf = df[['max','median','min','sd_stdDev','aspect','elevation','hillshade','slope','B9','B10']]
#Wholescale needed on this DF
nWscdf = df[['B1','B2','B3','B4','B5','B6','B7','B8','B11','B12']]

max=nWscdf.select_dtypes(include=[np.number]).values.max()
min=nWscdf.select_dtypes(include=[np.number]).values.min()
range=max-min



#MinMaxscaling

#Wholescaled DF
Wscdf = nWscdf.sub(min).divide(range)

#Scaling of Independend DF
rnames=list(scdf.index)
cnames=list(scdf)
scaler = MinMaxScaler()
scaler.fit(scdf)
scaled=scaler.transform(scdf)

rnames=scdf.index
cnames=list(scdf)
scaled=pd.DataFrame(scaled, index=rnames,columns=cnames)
print(scaled)

skal_vectoren = np.array(scaled.iloc[1:,2:])
C_vectoren = np.array(df[['C']])

model = Sequential([
					Dense(21, input_shape=(21,), activation='relu'),
					Dense(100, activation='relu'),
					Dense(30, activation='relu'),
					Dense(1, activation='softmax')
				   ])

model.compile(optimizer='adam', loss=mean_squared_error, metrics['accuracy'])

model.fit(skal_vectoren,C_vectoren, epochs=10, batch_size=15, validation_split=10)






