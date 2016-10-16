import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from skimage.filters import *
from skimage.feature import greycomatrix, greycoprops

Targets = np.genfromtxt("data/targets.csv")

X_train = []
for i in range(1, 279):
	example = nib.load("data/set_train/train_"+str(i)+".nii")
	image = example.get_data()
	I = image[:, :, 91, 0]
	I=skimage.filters.g
	np.asarray(I)
	Iflat = I.flatten(order='C')
	X_train.append(Iflat)
Data = X_train

X_train, X_test, y_train, y_test = \
		train_test_split(Data, Targets, test_size=0.33, random_state=42)

#convert data to feature selected data then feed into the SVM (pip_svr)

os.chdir('data/set_train')
L=os.listdir('./')

img = nib.load(L[0])
data = img.get_data()
data = data[:, :, 80, 0]

#scaling the data
min_max_scaler = preprocessing.MinMaxScaler()
scaled_data = min_max_scaler.fit_transform(data)

result = greycomatrix(scaled_data, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)

graph=result[:,:,0,3]
print np.amax(graph)

plt.matshow(graph)
plt.show()

#
# # Pipeline that scales (StandardScaler()), performes dimensionality reduction with PCA and trains a support vector
# # regression machine classifier.
# pipe_svr = Pipeline([('scl', StandardScaler()),
# 						('clf', SVR(kernel='linear', C=1.0))])
#
# pipe_svr.fit(X_train, y_train)
# print('Test Accuracy: %.3f' % pipe_svr.score(X_test, y_test))
