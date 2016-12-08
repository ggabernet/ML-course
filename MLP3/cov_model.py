import nibabel as nib
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV

from nibabel import processing

from scipy import ndimage

Targets = np.genfromtxt("data/targets.csv")

Image_dimension_2D=176*208
Number_of_images=278

slice_number=80


Image_dimension=Image_dimension_2D

X_train = []
X_data=np.zeros((Image_dimension,Number_of_images))

Target_data=[]

for i in range(1, Number_of_images+1):
	print "Current image is :", str(i)
	Target_data.append(Targets[i-1])
	example = nib.load("data/set_train/train_"+str(i)+".nii")
	image = example.get_data()
	I = image[:, :, slice_number, 0]
	I=np.asarray(I, dtype=float)
	I=I/np.max(I)
	Iflat=I.flatten(order='C')
	X_train.append(Iflat)
	X_data[:,i-1]=Iflat
Data = X_train

X_data = np.asarray(X_data)
Target_data=np.asarray(Target_data)

X_cov=np.zeros((Image_dimension))
X_corr = np.zeros((Image_dimension))

ignore_index=[]

threshold=0

print Target_data

for i in range(1,Image_dimension):
	print "processed cov:", i
	cov=np.mean(np.dot(X_data[i,:],Target_data))-np.mean(X_data[i,:])*np.mean(Target_data)
	corr = cov / (np.std(X_data[i,:]) * np.std(Target_data))
	if cov > threshold:
		X_cov[i]=cov
		X_corr[i] = cov / (np.std(X_data) * np.std(Target_data))
	else:
		ignore_index.append(i)

Data=np.delete(X_data, ignore_index ,axis=0)
Data=np.transpose(Data)

plt.imshow(X_corr.reshape(176,208))
plt.show()





