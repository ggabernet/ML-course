from skimage import measure
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel import processing as pr
import numpy as np
from sklearn.cross_validation import train_test_split
from feature_extraction import Contours
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

cnt = Contours(1000, 100)
contours = cnt.get_2D_contours(X_train)
print contours
