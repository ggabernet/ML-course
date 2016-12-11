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
import skimage
from skimage.filters import roberts, sobel, scharr, prewitt, gaussian
from sklearn.model_selection import GridSearchCV

from skimage import feature
from skimage import exposure
from nibabel import processing

from scipy import ndimage

Targets = np.genfromtxt("data/targets.csv")

Image_dimension_2D=176*208
Image_dimension_3D=176*208*176
Number_of_images=278

slice_number=80
Full_3D_image='False'

if Full_3D_image == 'False':
	Image_dimension=Image_dimension_2D
else:
	Image_dimension=Image_dimension_3D

fwhm = processing.sigma2fwhm(1)

X_train = []
X_data=np.zeros((Image_dimension,Number_of_images))

Target_data=[]

for i in range(1, Number_of_images+1):
	print "Current image is :", str(i)
	Target_data.append(Targets[i-1])
	example = nib.load("data/set_train/train_"+str(i)+".nii")
	example=processing.smooth_image(example, fwhm)
	image = example.get_data()
	if Full_3D_image == 'False':
		I = image[:, :, slice_number, 0]
	else:
		I = image[:, :, :, 0]
	I=np.asarray(I, dtype=float)
	#scale data
	#min_max_scaler = preprocessing.MinMaxScaler()
	#I = min_max_scaler.fit_transform(I)
	I=I/np.max(I)
	#Image processing
	#Ix=ndimage.prewitt(I, axis=1)
	I=prewitt(I)
	I = gaussian(I, sigma=1)
	Iflat=I.flatten(order='C')
	X_train.append(Iflat)
	X_data[:,i-1]=Iflat
Data = X_train

X_data = np.asarray(X_data)
Target_data=np.asarray(Target_data)

X_cov=np.zeros((Image_dimension))

ignore_index=[]

threshold=50

for i in range(1,Image_dimension):
	cov=np.mean(np.dot(X_data[i,:],Target_data))-np.mean(X_data[i,:])*np.mean(Target_data)
	if cov > threshold:
		X_cov[i]=cov
	else:
		ignore_index.append(i)

Data=np.delete(X_data, ignore_index ,axis=0)
Data=np.transpose(Data)

plt.imshow(X_cov.reshape(176,208))
plt.show()

X_train, X_test, y_train, y_test = \
		train_test_split(Data, Target_data, test_size=0.33, random_state=42)

#X_train, X_test, y_train, y_test =

clf=linear_model.Lasso(alpha=0.001)

# regression machine classifier.
regr = Pipeline([('scl', StandardScaler()),
						('pca', PCA(n_components=100)),
						('clf',clf)])

# Train the model using the training setskf=KF
regr.fit(X_train, y_train)

# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))

# Plot outputs

# # Pipeline that scales (StandardScaler()), performs dimensionality reduction with PCA and trains a support vector
# # regression machine classifier.
# pipe_svr = Pipeline([('scl', StandardScaler()),
# 						('pca', PCA(n_components=20)),
# 						('clf', SVR(kernel='linear', C=1))])
# pipe_svr.fit(X_train, y_train)
# print('R^2 score: %.3f' % pipe_svr.score(X_test, y_test))
#
# plt.scatter(y_train, pipe_svr.predict(X_train), color='Blue')
# plt.scatter(y_test, pipe_svr.predict(X_test), color='Red')

plt.scatter(y_train, regr.predict(X_train), color='Blue')
plt.scatter(y_test, regr.predict(X_test), color='Red')

axes = plt.gca()
axes.set_xlim([0,int(np.max(y_train)+10)])
axes.set_ylim([0,int(np.max(y_train)+10)])

plt.plot()
plt.show()

# true_test=[]
#
# for i in range(1, 139):
# 	example = nib.load("data/set_test/test_"+str(i)+".nii")
# 	image = example.get_data()
# 	I = image[:, :, 80, 0]
# 	I=np.asarray(I, dtype=float)
# 	#scale data
# 	min_max_scaler = preprocessing.MinMaxScaler()
# 	I = min_max_scaler.fit_transform(I)
# 	#Image processing
# 	I = prewitt(I)
# 	Iflat = I.flatten(order='C')
# 	true_test.append(Iflat)
#
#
# output=open('Submission.csv','w+')
# output.write("ID,Prediction"+'\n')
#
# for idx, line in enumerate(regr.predict(true_test)):
# 	  print line
# 	  output.write(str(idx+1)+','+str(line)+'\n')
# output.close()






