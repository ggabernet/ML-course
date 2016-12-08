from skimage import measure
import numpy as np
import nibabel as nib
from feature_extraction import CenterCutCubes, PvalSelect, Select
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from scipy import ndimage

from sklearn import linear_model
import matplotlib.pyplot as plt

Targets = np.genfromtxt("data/targets.csv")

Data = []
for i in range(1, 279):
    imagefile = nib.load("data/set_train/train_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    I = np.asarray(I, dtype='float')
    I = ndimage.prewitt(I, axis=0)
    I = ndimage.gaussian_filter(I, sigma=1)
    imagefile.uncache()
    Data.append(np.asarray(I))

Data = np.asarray(Data)

X_train, X_test, y_train, y_test = \
    train_test_split(Data, Targets, test_size=0.33, random_state=42, stratify=Targets)

#X_train = Data
#y_train = Targets

print "fitting has started"

clf = linear_model.LogisticRegression()
#svm_model = SVC(C=0.0001, kernel='linear', class_weight={0: 6, 1: 1}, probability=True)

pipe = Pipeline([('cut', CenterCutCubes(size_cubes=5, plane_jump=1, x1=50, y1=80, z1=50, x2=120, y2=150, z2=100)),
                ('var', VarianceThreshold()),
                ('sel', Select(type='mutual_info', threshold=0.1)),
                ('scl', StandardScaler()),
                ('pca', PCA(n_components=5)),
                ('clf', clf)
                ])

pipe.fit(X_train, y_train)


print "Train - log loss :", log_loss(y_train,pipe.predict_proba(X_train))
print "Test - log loss :", log_loss(y_test,pipe.predict_proba(X_test))

##############################
#      Submission.csv        #
##############################

Data_test = []
for i in range(1, 139):
    print "Test image ", str(i), "processed"
    imagefile = nib.load("data/set_test/test_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    I = np.asarray(I, dtype='float')
    I = ndimage.prewitt(I, axis=0)
    I = ndimage.gaussian_filter(I, sigma=1)
    #I = I.flatten(order='C')
    Data_test.append(np.asarray(I))

X_test = Data_test

predictions = pipe.predict_proba(X_test)

with open("final_sub.csv", mode='w') as f:
    f.write("ID,Prediction\n")
    for idx, pred in enumerate(predictions):
        f.write(str(idx+1)+','+str(pred[1])+'\n')
