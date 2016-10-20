import nibabel as nib
from nibabel import processing as pr
import numpy as np
from sklearn.cross_validation import train_test_split
from feature_extraction import Contours, Intensities
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import *
from sklearn.linear_model import Lasso
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR
from sklearn.decomposition import PCA

Targets = np.genfromtxt("data/targets.csv")

Data = []
for i in range(1, 279):
    imagefile = nib.load("data/set_train/train_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    Data.append(I)

X_train, X_test, y_train, y_test = \
    train_test_split(Data, Targets, test_size=0.33, random_state=42)

cnt = Intensities(layers_x_dim=11)
cnt.calculate_descriptor(X_train)
desc_train = cnt.descriptor
cnt.calculate_descriptor(X_test)
desc_test = cnt.descriptor

print desc_train.shape
np.savetxt("intensities_descriptor.csv",desc_train, delimiter=',')