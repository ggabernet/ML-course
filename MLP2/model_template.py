from skimage import measure
import numpy as np
import nibabel as nib
from feature_extraction import CenterCutCubes, CovSel, CenterCut, Filtering, Covariance
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import *
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA, RandomizedPCA

from sklearn import linear_model
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from skimage.filters import roberts, sobel, scharr, prewitt, gaussian
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from nibabel import processing
from scipy import ndimage

#############################################
#       Data loading and preprocessing      #
#############################################

Targets = np.genfromtxt("data/targets.csv")

[x_low, x_up] = [50, 120]
[y_low, y_up] = [80, 150]
[z_low, z_up] = [50, 100]

Data = []
for i in range(1, 279):
    print "Train image ", str(i), "processed"
    imagefile = nib.load("data/set_train/train_"+str(i)+".nii")
    imagefile = processing.smooth_image(imagefile, 1)
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    I=I[x_low:x_up, y_low:y_up]
    I = np.asarray(I, dtype=float)
    I = I/np.max(I)
    imagefile.uncache()
    I = I.flatten(order='C')
    Data.append(np.asarray(I))


X_train = Data
y_train = Targets

print np.asarray(X_train).shape

################################
#       Data set splitting     #
################################

#the inital input data
X_train = Data
y_train = Targets

#setting the empty lists
Data_class_0 = []
Data_class_1 = []
Targets_class_0 = []
Targets_class_1 = []

#seperates the data sets into class_0 and class_1
#also saves the index for these data sets in Targets_class_0 and Targets_class_1
for i in range(0, len(Targets)):
    if Targets[i] == 0:
        #stores the data belonging to class 0
        Data_class_0.append(Data[i])
        #saves the index for the class 0 data sets
        Targets_class_0.append(Targets[i])
    if Targets[i] == 1:
        # stores the data belonging to class 1
        Data_class_1.append(Data[i])
        # saves the index for the class 1 data sets
        Targets_class_1.append(Targets[i])

#the percent of data sets in the training set
percent_train = 0.33

#saves the first (percent_train) indices for use in the training set for both class 0 and 1 data sets
train_set_0 = range(0, int(percent_train*len(Targets_class_0)))
train_set_1 = range(0, int(percent_train*len(Targets_class_1)))

#saves the remaining (percent_train) indices for use in the testing set for both class 0 and 1 data sets
test_set_0 = set(range(0,len(Targets_class_0))).difference(set(train_set_0))
test_set_1 = set(range(0,len(Targets_class_1))).difference(set(train_set_1))

X_train = []
y_train = []

#train set

for i in train_set_0:
    X_train.append(Data_class_0[i])
    y_train.append(Targets_class_0[i])
for i in train_set_1:
    X_train.append(Data_class_1[i])
    y_train.append(Targets_class_1[i])

#test set

X_test = []
y_test = []

for i in test_set_0:
    X_test.append(Data_class_0[i])
    y_test.append(Targets_class_0[i])
for i in test_set_1:
    X_test.append(Data_class_1[i])
    y_test.append(Targets_class_1[i])

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

##############################
#         Fit model          #
##############################

#X_train, X_test, y_train, y_test = train_test_split(Data, Targets, test_size=0.33,random_state=42)

clf = linear_model.LogisticRegression()

best_pipe = Pipeline([('scl', StandardScaler()),
                  ('var', VarianceThreshold()),
                  ('pca', PCA(n_components=5)),
                  ('clf', clf)])

best_pipe.fit(X_train, y_train)

print "Log Loss :", log_loss(y_train,best_pipe.predict_proba(X_train), normalize='False', labels=[0,1])
print "Log Loss :", log_loss(y_test,best_pipe.predict_proba(X_test),normalize='False', labels=[0,1])


##############################
#      Submission.csv        #
##############################

Data_test = []
for i in range(1, 139):
    print "Test image ", str(i), "processed"
    imagefile = nib.load("data/set_test/test_"+str(i)+".nii")
    imagefile = processing.smooth_image(imagefile, 1)
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    I = I[x_low:x_up, y_low:y_up]
    I = I/np.max(I)
    I=I.flatten(order='C')
    Data_test.append(np.asarray(I))

X_test = Data_test

predictions = best_pipe.predict_proba(X_test)

with open("Submission_fixed.csv", mode='w') as f:
    f.write("ID,Prediction\n")
    for idx, pred in enumerate(predictions):
        f.write(str(idx+1)+','+str(pred[1])+'\n')