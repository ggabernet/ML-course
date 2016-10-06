import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np

y_train = np.genfromtxt("targets.csv")

X_train=[]
for i in range(1, 279):
	example = nib.load("set_train/train_"+str(i)+".nii")
	image = example.get_data()
	I = image[:, :, 91, 0]
	np.asarray(I)
	Iflat = I.flatten(order='C')
	X_train.append(Iflat)
X_train


# --------- SVM model ---------------------------
svm = SVC(kernel='linear', C=1.0,random_state=0)
svm.fit(X_train, y_train)
svm.score(X_train, y_train)
