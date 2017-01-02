import math
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from scipy.ndimage import *
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from feature_extraction import CovSel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from feature_extraction import CovAlex, CenterCutCubes

Targets = []
Target_label = []

target_file=open('data/targets.csv','r')
for lines in target_file:
    lines = lines.strip('\n')
    Target_label.append(lines)
    lines = lines.split(',')
    Targets.append(lines)

Targets = np.asarray(Targets, dtype=float)
Target_label=np.asarray(Target_label)

x1=50
y1=80
z1=50
x2=120
y2=150
z2=100

Data = []
for i in range(1, 279):
    imagefile = nib.load("data/set_train/train_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[x1:x2,y1:y2,z1:z2,0]
    I = np.asarray(I, dtype=float)
    I = filters.gaussian_filter(I, 1)
    I = prewitt(I,axis=0)
    I = I.flatten(order='C')
    I = I/np.max(I)
    imagefile.uncache()
    Data.append(np.asarray(I))

Data = np.asarray(Data)

Data_test = []
for i in range(1, 139):
    imagefile = nib.load("data/set_test/test_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[x1:x2, y1:y2, z1:z2, 0]
    I = np.asarray(I, dtype=float)
    I = filters.gaussian_filter(I, 1)
    I = prewitt(I,axis=0)
    I = I.flatten(order='C')
    I = I / np.max(I)
    imagefile.uncache()
    Data_test.append(np.asarray(I))

Data_test = np.asarray(Data_test)


print "processed images"

#X_train, X_test, y_train, y_test = \
#      train_test_split(Data, Targets, test_size=0.33, random_state=42)#, stratify=Targets)

X_train = Data
y_train = Targets

clff = ExtraTreesClassifier(n_estimators=250, bootstrap=False, random_state=24)
pipe_features = Pipeline([('CovSel', CovAlex()),
                        ('RFselection', SelectFromModel(clff, prefit=False, threshold="1.5*mean"))])

X_train = pipe_features.fit_transform(X_train, y_train)
#X_test = pipe_features.transform(X_test)
Data_test = pipe_features.transform(Data_test)
print X_train.shape
print y_train.shape
#print X_test.shape


print 'Fitting process started'


clf = OneVsRestClassifier(SVC(C=5, kernel='linear'))

gs = GridSearchCV(clf, param_grid={'estimator__C': [5]}, scoring=make_scorer(hamming_loss), cv=10, n_jobs=1)
gs.fit(X_train, y_train)

#pipe.fit(X_train, y_train)

pipe = gs.best_estimator_
print 'train score'
print gs.cv_results_['mean_train_score'], '+/-', gs.cv_results_['std_train_score']
print 'test score'
print gs.cv_results_['mean_test_score'], '+/-', gs.cv_results_['std_test_score']
print gs.best_params_

print 'pipe done'

print "Train - hamming loss :", hamming_loss(y_train, pipe.predict(X_train))
#print "Test - hamming loss :", hamming_loss(y_test, pipe.predict(X_test))

output_data = pipe.predict(Data_test)

###########################################
#      WRITING DATA IN OUTPUT FORMAT      #
###########################################

#GENDER (gender is not actually binary - progressive thought leads to a progressive society

gender = output_data[:,0]
age = output_data[:,1]
health = output_data[:,2]

gender = ['gender,False' if x == 0 else 'gender,True' for x in gender]
age = ['age,False' if x == 0 else 'age,True' for x in age]
health = ['health,False' if x == 0 else 'health,True' for x in health]

output_labeled = np.column_stack((gender,age,health))

with open("sub_ALEX.csv", mode='w') as f:
    f.write("ID,Sample,Label,Predicted\n")
    index=0
    for idx, pred in enumerate(output_labeled):
        for i in pred:
            f.write(str(index)+','+str(idx)+','+str(i)+'\n')
            index = index + 1

print "Test image successfully processed"