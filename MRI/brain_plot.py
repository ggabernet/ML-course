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


Targets = np.genfromtxt("data/targets.csv")

X_train = []
for i in range(1, 279):
	example = nib.load("data/set_train/train_"+str(i)+".nii")
	image = example.get_data()
	I = image[:, :, 80, 0]
	I=np.asarray(I, dtype=float)
	Iflat = I.flatten(order='C')
	X_train.append(Iflat)
Data = X_train

X_train, X_test, y_train, y_test = \
		train_test_split(Data, Targets, test_size=0.33, random_state=42)

regr = linear_model.Lasso(alpha=200)

# Train the model using the training sets
regr.fit(X_train, y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
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

true_test=[]

for i in range(1, 139):
	example = nib.load("data/set_test/test_"+str(i)+".nii")
	image = example.get_data()
	I = image[:, :, 80, 0]
	np.asarray(I)
	I=I/np.max(I)
	Iflat = I.flatten(order='C')
	true_test.append(Iflat)



output=open('Submission.csv','w')
output.write("ID, Preidcted_ages"+'\n')
index=1
for line in regr.predict(true_test):
	  output.write(str(index)+','+str(line)+'\n')
	  index=index+1
output.close()

plt.show()





