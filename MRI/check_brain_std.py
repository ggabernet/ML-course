import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

y_train = np.genfromtxt("targets.csv")

for n in range(1, 176, 10):
	X_train=[]
	for i in range(1, 279):
		example = nib.load("set_train/train_"+str(i)+".nii")
		image = example.get_data()
		I = image[:, :, 91, 0]
		np.asarray(I)
		Iflat = I.flatten(order='C')
		X_train.append(Iflat)
	X_train = np.array(X_train)
	std_mat = np.std(X_train, axis=0)
	print std_mat
	print "hello"
	plt.matshow(std_mat)
	plt.savefig("output/std/std_mat%d.png" %n)