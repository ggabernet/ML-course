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


ccc = CenterCutCubes(size_cubes=3, plane_jump=3, y1=20, x1=20, z1=20, x2=150, y2=180, z2=150)
ccc.fit(Data[:100])
Data_ccc = ccc.transform(Data[:100])
Data_ccc = np.array(Data_ccc)
print Data_ccc.shape
#
v=VarianceThreshold(1)
v.fit(Data_ccc)
Data_ccc=v.transform(Data_ccc)
print Data_ccc.shape
#
s=Select(type="mutual_info",threshold=0.00001)
s.fit(Data_ccc[:100],Targets[:100])
Data_ccc=s.transform(Data_ccc)
#
print Data_ccc.shape
#


X_train, X_test, y_train, y_test = \
     train_test_split(Data, Targets, test_size=0.33, random_state=42)

#mut_inf = mutual_info_classif(np.array(X_train_ccc), y_train, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)


pipe = Pipeline([('cut', CenterCutCubes(size_cubes=2)),
                 ('var', VarianceThreshold(1)),
                 ('sel', Select(type="f_value",threshold=0.001)),
                 ('scl', StandardScaler()),
                 ('pca', PCA( n_components=100)),
                 ('clf', SVC(kernel="linear",degree=2))])


gs = GridSearchCV(estimator=pipe,
                   param_grid=[{'cut__size_cubes': [3],
                                'cut__y1': [10],
                                'cut__x1': [10],
                                'cut__z1': [10],
                                'cut__x2': [160],
                                'cut__y2': [190],
                                'cut__z2': [160],
                                #'clf__n_estimators': [10]}],
                                'pca__n_components': [100],
                                'sel__type': ["f_value","chi2","mutual_info"],
                                'clf__kernel': ["linear","poly"],
                                'clf__degree': [2],
                                'clf__C': [0.01]}],
                   error_score=999,
                   cv=5,
                   n_jobs=1,
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

