from skimage import measure
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

Targets = np.genfromtxt("data/targets.csv")

X_train = []
for i in range(1, 279):
	example = nib.load("data/set_train/train_"+str(i)+".nii")
	image = example.get_data()
	I = image[:, :, 90, 0]
	np.asarray(I)
	Iflat = I.flatten(order='C')
	X_train.append(Iflat)
Data = X_train

X_train, X_test, y_train, y_test = \
		train_test_split(Data, Targets, test_size=0.33, random_state=42)

X_train = np.asarray(X_train)
pipe_scale = Pipeline([('scl', MinMaxScaler())])
pipe_scale.fit(X_train, y_train)
X_trans = pipe_scale.transform(X_train)

X_img = []
for n in X_trans:
	X_img.append(n.reshape(I.shape,order='C'))
X_img = np.asarray(X_img)

for n in range(1, 13):
	contours = measure.find_contours(X_img[n], 0.55)
	bigcontours = []
	for c in contours:
		if c.shape[0] > 100:
			bigcontours.append(c)


	fig, ax = plt.subplots()
	ax.imshow(X_img[n], interpolation='nearest', cmap=plt.cm.gray)

	for n, contour in enumerate(bigcontours):
		ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

	ax.axis('image')
	ax.set_xticks([])
	ax.set_yticks([])
	plt.show()