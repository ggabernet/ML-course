import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split

Targets = np.genfromtxt("data/targets.csv")

X_train = []
for i in range(1, 279):
	example = nib.load("data/set_train/train_"+str(i)+".nii")
	image = example.get_data()
	I = image[:, :, 91, 0]
	np.asarray(I)
	Iflat = I.flatten(order='C')
	X_train.append(Iflat)
Data = X_train

X_train, X_test, y_train, y_test = \
		train_test_split(Data, Targets, test_size=0.33, random_state=42)


# Pipeline that scales (StandardScaler()), performs dimensionality reduction with PCA and trains a support vector
# regression machine classifier.
pipe_svr = Pipeline([('scl', StandardScaler()),
						('pca', PCA(n_components=100)),
						('clf', SVR(kernel='linear', C=1.0))])
pipe_svr.fit(X_train, y_train)
print('Test Accuracy: %.3f' % pipe_svr.score(X_test, y_test))
