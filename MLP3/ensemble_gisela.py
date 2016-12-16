from skimage import measure
import numpy as np
import nibabel as nib
from feature_extraction import CenterCutCubes, PvalSelect, Select
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import hamming_loss
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier\

from scipy import ndimage
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier


Targets = np.genfromtxt("data/targets.csv", delimiter=',')

Targets = np.asarray(Targets)

Data = []
for i in range(1, 279):
    imagefile = nib.load("data/set_train/train_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    imagefile.uncache()
    Data.append(np.asarray(I))

print I.shape
Data = np.asarray(Data)

X_train, X_test, y_train, y_test = \
      train_test_split(Data, Targets, test_size=0.33, random_state=42)

# X_train = Data
# y_train = Targets

sex_target = y_train[:, 0]
age_target = y_train[:, 1]
health_target = y_train[:, 2]




# ------------
# Sex model
# ------------
print 'Performing grid search sex'

pipe = Pipeline([('cut', CenterCutCubes(size_cubes=5, plane_jump=2, x1=20, y1=20, z1=20, x2=150, y2=180, z2=150)),
                 ('var', VarianceThreshold()),
                 ('sel', Select(type='mutual_info', threshold=0.1)),
                 ('scl', StandardScaler()),
                 # ('pca', PCA(n_components=50)),
                 ('clf', SVC())])

gs_sex = GridSearchCV(pipe,
                    param_grid={'clf__C' : [0.001, 0.01, 0.1],
                                'clf__kernel': ['linear','rbf'],
                                'clf__gamma': [0.01, 0.1, 1]},
                    scoring=make_scorer(matthews_corrcoef),
                    n_jobs = -1)

gs_sex.fit(X_train, sex_target)

sex_model = gs_sex.best_estimator_

print "Train - hamming loss :", hamming_loss(sex_target, gs_sex.predict(X_train))
print "Test - hamming loss :", hamming_loss(y_test[:, 0], gs_sex.predict(X_test))

# ------------
# Age model
# ------------

print 'Performing grid search age'

pipe = Pipeline([('cut', CenterCutCubes(size_cubes=5, plane_jump=2, x1=20, y1=20, z1=20, x2=150, y2=180, z2=150)),
                 ('var', VarianceThreshold()),
                 ('sel', Select(type='mutual_info', threshold=0.1)),
                 ('scl', StandardScaler()),
                 # ('pca', PCA(n_components=50)),
                 ('clf', SVC())])

gs_age = GridSearchCV(pipe,
                      param_grid={'clf__C' : [0.001, 0.01, 0.1],
                                    'clf__kernel': ['linear','rbf'],
                                    'clf__gamma': [0.01, 0.1, 1]},
                      scoring=make_scorer(matthews_corrcoef),
                      n_jobs=-1)

gs_age.fit(X_train, age_target)

age_model = gs_age.best_estimator_

print "Train - hamming loss :", hamming_loss(age_target, gs_age.predict(X_train))
print "Test - hamming loss :", hamming_loss(y_test[:, 1], gs_age.predict(X_test))

# ------------
# Health model
# ------------

print 'Performing grid search health'

pipe = Pipeline([('cut', CenterCutCubes(size_cubes=5, plane_jump=2, x1=20, y1=20, z1=20, x2=150, y2=180, z2=150)),
                 ('var', VarianceThreshold()),
                 ('sel', Select(type='mutual_info', threshold=0.1)),
                 ('scl', StandardScaler()),
                 # ('pca', PCA(n_components=50)),
                 ('clf', SVC())])

gs_health = GridSearchCV(pipe,
                         param_grid={'clf__C' : [0.001, 0.01, 0.1],
                                     'clf__kernel': ['linear','rbf'],
                                     'clf__gamma': [0.01, 0.1, 1]},
                         scoring=make_scorer(matthews_corrcoef),
                         n_jobs=-1)

gs_health.fit(X_train, health_target)
health_model = gs_health.best_estimator_

print "Train - hamming loss :", hamming_loss(health_target, gs_health.predict(X_train))
print "Test - hamming loss :", hamming_loss(y_test[:, 2], gs_health.predict(X_test))

# ----------------------------------
# Calculating scores for all classes
# ----------------------------------
y_all_train = np.hstack((sex_model.predict(X_train), age_model.predict(X_train), health_model.predict(X_train)))
y_all_test = np.hstack((sex_model.predict(X_test), age_model.predict(X_test), health_model.predict(X_test)))

print "Train - hamming loss :", hamming_loss(y_train.flatten(order='C'), y_all_train.flatten(order='C'))
print "Test - hamming loss :", hamming_loss(y_test.flatten(order='C'), y_all_test.flatten(order='C'))

Data_test = []
for i in range(1, 139):
    imagefile = nib.load("data/set_test/test_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    imagefile.uncache()
    Data_test.append(np.asarray(I))



###########################################
#      WRITING DATA IN OUTPUT FORMAT      #
###########################################

#GENDER (gender is not actually binary - progressive thought leads to a progressive society

gender = sex_model.predict(Data_test)
age = age_model.predict(Data_test)
health = health_model.predict(Data_test)

gender = ['gender,False' if x == 0 else 'gender,True' for x in gender]
age = ['age,False' if x == 0 else 'age,True' for x in age]
health = ['health,False' if x == 0 else 'health,True' for x in health]

output_labeled = np.column_stack((gender, age, health))

with open("ensemble_gisela.csv", mode='w') as f:
    f.write("ID,Sample,Label,Predicted\n")
    index = 0
    for idx, pred in enumerate(output_labeled):
        for i in pred:
            f.write(str(index)+','+str(idx)+','+str(i)+'\n')
            index = index + 1

print "Test image successfully processed"