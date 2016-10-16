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
Data = np.asarray(X_train, dtype=float)

X_train, X_test, y_train, y_test = \
		train_test_split(Data, Targets, test_size=0.33, random_state=42)

#data binning
bins=np.linspace(0,1,1000)
X_train=X_train/np.max(X_train)
X_train=np.digitize(X_train,bins)

for n in range(1, 13):
	contours = measure.find_contours(X_train[n], 600)
	bigcontours = []
	c_index=1
	print "Image", n, "Age", Targets[n]
	for c in contours:
		if c.shape[0] > 100:
			bigcontours.append(c)
			x_coord = np.asarray(c[:,0])
			y_coord = np.asarray(c[:,1])
			y_next=np.roll(y_coord,-1)
			y_diff=[abs(x) for x in (y_next-y_coord)]

			Area_contour = np.sum(np.prod([x_coord, y_diff],axis=0))

			#print np.sum(np.multiply(x_coord,y_diff))


			print 'Contour', c_index, ' :', Area_contour



			c_index=c_index+1

	fig, ax = plt.subplots()
	ax.imshow(X_train[n], interpolation='nearest', cmap=plt.cm.gray)

	for n, contour in enumerate(bigcontours):
		ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

	ax.axis('image')
	ax.set_xticks([])
	ax.set_yticks([])
	plt.show()