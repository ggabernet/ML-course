from skimage import measure
import numpy as np
import nibabel as nib
from feature_extraction import CenterCutCubes, Select
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import mutual_info_classif, f_classif

Targets = np.genfromtxt("data/targets.csv")

Data = []
for i in range(1, 279):
    imagefile = nib.load("data/set_train/train_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    imagefile.uncache()
    Data.append(np.asarray(I))

print I.shape

# -------------------------
# SPLITTING TRAIN/TEST SETS
#--------------------------

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

X_test = np.asarray(X_test)
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

pipe = Pipeline([('cut', CenterCutCubes(size_cubes=2, plane_jump=2, x1=30, y1=30, z1=30, x2=150, y2=170, z2=140)),
                 ('var', VarianceThreshold(1)),
                 ('sel', Select(type="f_value",threshold=0.1)),
                 ('scl', StandardScaler()),
                 ('pca', PCA( n_components=100)),
                 ('clf', SVC(kernel="linear",degree=2))])


gs = GridSearchCV(estimator=pipe,
                   param_grid=[{'pca__n_components': [100],
                                'sel__type': ["f_value"],
                                'clf__kernel': ["linear"],
                                'clf__C': [0.01]}],
                   error_score=999,
                   cv=5,
                   n_jobs=-1,
                   verbose=10,
                   scoring=make_scorer(log_loss))

gs.fit(X_train, y_train)

best_pipe = gs.best_estimator_
print gs.best_params_

means = gs.cv_results_['mean_test_score']
stds = gs.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

best_pipe.fit(X_train, y_train)


y_test_predicted = best_pipe.predict(X_test)
score_test = log_loss(y_test, y_test_predicted)
print("log_loss test data: " + str(score_test))


y_train_predicted = best_pipe.predict(X_train)
score_train = log_loss(y_train, y_train_predicted)
print("log_loss train data: " + str(score_train))


Data_test = []
for i in range(1, 139):
    imagefile = nib.load("data/set_test/test_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    Data_test.append(np.asarray(I))

# Data_ccc = ccc.transform(Data_test)
# Data_ccc = np.array(Data_ccc)
# print Data_ccc.shape
#
# Data_ccc=v.transform(Data_ccc)
# Data_test=s.transform_test(Data_ccc)

X_test = Data_test
predictions = best_pipe.predict(X_test)

with open("Scores.csv", mode='w') as f:
    f.write("ID,Prediction\n")
    for idx, pred in enumerate(predictions):
        pred = round(pred, 0)
        f.write(str(idx+1)+','+str(pred)+'\n')

