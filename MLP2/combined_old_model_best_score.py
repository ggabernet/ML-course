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

X_train, X_test, y_train, y_test = \
     train_test_split(Data, Targets, test_size=0.33, random_state=42)


ccc = CenterCutCubes(size_cubes=5, plane_jump=1,  y1=10, x1=10, z1=10, x2=170, y2=170, z2=200)
ccc.fit(X_train)
X_train = ccc.transform(X_train)
X_train = np.array(X_train)
X_test = ccc.transform(X_test)
X_test = np.array(X_test)
print X_train.shape
print X_test.shape
#
v=VarianceThreshold()
v.fit(X_train)
X_train=v.transform(X_train)
X_test=v.transform(X_test)
print X_train.shape
print X_test.shape
#
sc=StandardScaler()
sc.fit(X_train)
X_train=sc.transform(X_train)
X_test=sc.transform(X_test)
#
pc=PCA(n_components=250)
pc.fit(X_train)
X_train=pc.transform(X_train)
X_test=pc.transform(X_test)
print X_train.shape
print X_test.shape


pipe = Pipeline([#('cut', CenterCutCubes(size_cubes=3, plane_jump=3)),
                 #('var', VarianceThreshold(1)),
                 #('sel', Select(type="f_value",threshold=0.1)),
                 #('scl', StandardScaler()),
                 #('pca', PCA( n_components=100)),
                 ('clf', SVC(C=0.1, kernel="linear"))])


gs = GridSearchCV(estimator=pipe,
                   param_grid=[{#'cut__size_cubes': [3],
                                #'cut__y1': [30],
                                #'cut__x1': [30],
                                #'cut__z1': [30],
                                #'cut__x2': [150],
                                #'cut__y2': [170],
                                #'cut__z2': [140],
                                #'clf__n_estimators': [10,50,100]}],
                                #'pca__n_components': [10],
                                #'sel__type': ["f_value"],
                                'clf__kernel': ["linear"],
                                'clf__C': [0.1]}],
                   error_score=999,
                   cv=5,
                   n_jobs=1,
                   verbose=10,
                   scoring=make_scorer(matthews_corrcoef))

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


X_test = Data_test
X_test = ccc.transform(X_test)
X_test = np.array(X_test)
#
X_test=v.transform(X_test)
#
X_test=sc.transform(X_test)
#
X_test=pc.transform(X_test)



predictions = best_pipe.(X_test)

with open("Scores.csv", mode='w') as f:
    f.write("ID,Prediction\n")
    for idx, pred in enumerate(predictions):
        pred = round(pred, 0)
        f.write(str(idx+1)+','+str(pred)+'\n')

