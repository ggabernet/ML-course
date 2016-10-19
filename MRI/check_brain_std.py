import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

y_train = np.genfromtxt("data/targets.csv")


X_train=[]
for i in range(1, 279):
    example = nib.load("data/set_train/train_"+str(i)+".nii")
    image = example.get_data()
    I = image[:, :, 91, 0]
    np.asarray(I)
    Iflat = I.flatten(order='C')
    X_train.append(Iflat)
X_train = np.array(X_train)
std_mat = np.std(X_train, axis=0)
std_mat = std_mat.reshape(I.shape, order='C')
plt.matshow(std_mat)
plt.savefig("output/std/std_mat.png")