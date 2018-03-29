#script for preparing datasets, loading fer2013 data and generating scaled images
import pandas as pd
import numpy as np

#loading fer2013.csv

data = pd.read_csv('fer2013.csv')
data = data['pixels']
data = [dat.split() for dat in data]
data = np.array(data)
data = data.astype('float64')
data = [[np.divide(d, 255.0) for d in dat] for dat in data]

np.save('/home/saif_m_dhrubo/uRECO/FacialExpressionClassifier/data/Scaled.bin.npy',data)
