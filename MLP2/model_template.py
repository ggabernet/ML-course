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

#############################################
#       Data loading and preprocessing      #
#############################################

Targets = np.genfromtxt("data/targets.csv")

[x_low, x_up] = [70, 160]
[y_low, y_up] = [65, 120]
[z_low, z_up] = [70, 50]

Data = []
for i in range(1, 279):
    imagefile = nib.load("data/set_train/train_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    #I=I[x_low:x_up, y_low:y_up]
    I = np.asarray(I, dtype=float)
    #I = gaussian(I, sigma=0.5)
    imagefile.uncache()
    I = I / np.max(I)
    I = I.flatten(order='C')
    Data.append(np.asarray(I))


X_train = Data
y_train = Targets

print np.asarray(X_train).shape

################################
#       Data set splitting     #
################################

X_train = Data
y_train = Targets

#X_train, X_test, y_train, y_test = \
#     train_test_split(Data, Targets, test_size=0.33, random_state=42)

Data_class_0 = []
Data_class_1 = []
Targets_class_0 = []
Targets_class_1 = []

for i in range(0, len(Targets)):
    if Targets[i] == 0:
        Data_class_0.append(Data[i])
        Targets_class_0.append(Targets[i])
    if Targets[i] == 1:
        Data_class_1.append(Data[i])
        Targets_class_1.append(Targets[i])

percent_train = 0.33

train_set_0 = range(0, int(percent_train*len(Targets_class_0)))
train_set_1 = range(0, int(percent_train*len(Targets_class_0)))

test_set_0 = set(range(0,len(Targets_class_0))).difference(set(train_set_0))
test_set_1 = set(range(0,len(Targets_class_1))).difference(set(train_set_1))

X_train = []
y_train = []

#train set

for i in train_set_0:
    X_train.append(Data_class_0[i][0])
    y_train.append(Targets_class_0[i])
for i in train_set_1:
    X_train.append(Data_class_1[i][0])
    y_train.append(Targets_class_1[i])

#test set

X_test = []
y_test = []

for i in test_set_0:
    X_test.append(Data_class_0[i][0])
    y_test.append(Targets_class_0[i])
for i in test_set_1:
    X_test.append(Data_class_1[i][0])
    y_test.append(Targets_class_1[i])


###plot class variation
Data_class_0 = np.asarray(Data_class_0)
print Data_class_0.shape

var_0 = np.mean(Data_class_0, axis=0)
print var_0.shape
plt.imshow(var_0)
plt.show()


Data_class_1 = np.asarray(Data_class_1)
print Data_class_1.shape

var_1 = np.mean(Data_class_1, axis=0)
print var_1.shape
plt.imshow(var_1)
plt.show()


##############################
#         Fit model          #
##############################

best_pipe = Pipeline([('scl', StandardScaler()),
                  ('var', VarianceThreshold()),
                  ('pca', PCA(n_components=10)),
                  ('clf', SVC(kernel='linear', C=0.1))])

best_pipe.fit(X_train, y_train)

print "Log Loss :", log_loss(y_train,best_pipe.predict(X_train))
print "Log Loss :", log_loss(y_test,best_pipe.predict(X_test))


##############################
#      Submission.csv        #
##############################

Data_test = []
for i in range(1, 139):
    imagefile = nib.load("data/set_test/test_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[80, :, :, 0]
    I=I[x_low:x_up, y_low:y_up]
    I = np.asarray(I, dtype=float)
    I = gaussian(I, sigma=0.5)
    I = I/np.max(I)
    I=I.flatten(order='C')
    Data_test.append(np.asarray(I))

X_test = Data_test

predictions = best_pipe.predict(X_test)

with open("SubmissionOptimizedCombinedModel3.csv", mode='w') as f:
    f.write("ID,Prediction\n")
    for idx, pred in enumerate(predictions):
        pred = round(pred,0)
        f.write(str(idx+1)+','+str(pred)+'\n')
