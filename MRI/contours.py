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
	X_train.append(I)
Data = X_train

X_train, X_test, y_train, y_test = \
		train_test_split(Data, Targets, test_size=0.33, random_state=42)

for n in range(1, 13):
	contours = measure.find_contours(X_train[n], 900)
	bigcontours = []
	for c in contours:
		if c.shape[0] > 100:
			bigcontours.append(c)

	fig, ax = plt.subplots()
	ax.imshow(X_train[n], interpolation='nearest', cmap=plt.cm.gray)

	for n, contour in enumerate(bigcontours):
		ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

	ax.axis('image')
	ax.set_xticks([])
	ax.set_yticks([])
	plt.show()