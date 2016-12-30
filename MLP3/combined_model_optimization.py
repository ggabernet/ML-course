from skimage import measure
import numpy as np
import nibabel as nib
from feature_extraction import CenterCutCubes
from feature_extraction_leon import Select
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

Targets = []
target_file=open('data/targets.csv','r')
for lines in target_file:
    lines = lines.strip('\n')
    lines = lines.split(',')
    Targets.append([int(l) for l in lines])

print len(Targets)

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


ccc = CenterCutCubes(size_cubes=3, plane_jump=2, x1=50, x2=120, y1=50, y2=150, z1=50, z2=100)
ccc.fit(X_train)
X_train = ccc.transform(X_train)
X_train = np.array(X_train)
X_test = ccc.transform(X_test)
X_test = np.array(X_test)
print "Cubes"
print X_train.shape
print X_test.shape
#
v=VarianceThreshold()
v.fit(X_train)
X_train=v.transform(X_train)
X_test=v.transform(X_test)
print "Variance"
print X_train.shape
print X_test.shape
#
sc=StandardScaler()
sc.fit(X_train)
X_train=sc.transform(X_train)
X_test=sc.transform(X_test)
#
#sel=Select(type='mutual_info', threshold=0.001)
#sel.fit(X_train, np.array(y_train))
#X_train=sel.transform(X_train)
#X_test=sel.transform(X_test)
#print "Mutual Info"
#print X_train.shape
#print X_test.shape
#
clff = ExtraTreesClassifier()
clff = clff.fit(X_train, np.array(y_train))
model = SelectFromModel(clff, prefit=True)
X_train = model.transform(X_train)
X_test = model.transform(X_test)
print X_train.shape
print X_test.shape
#
#pc=PCA(n_components=10)
#pc.fit(X_train)
#X_train=pc.transform(X_train)
#X_test=pc.transform(X_test)
#print X_train.shape
#print X_test.shape

y_train=np.array(y_train)


pipe = Pipeline([#('cut', CenterCutCubes(size_cubes=10)),
                #('scl', StandardScaler()),
                #('var', VarianceThreshold()),
                ('clf', OneVsRestClassifier(SVC(kernel="linear", C=0.001)))])


gs = GridSearchCV(estimator=pipe,
                  param_grid=[{}],#'cut__size_cubes': [10],
                               #'cut__y1': [50],
                               #'cut__x1': [50],
                               #'cut__z1': [50],
                               #'cut__x2': [170],
                               #'cut__y2': [200],
                               #'cut__z2': [170],
                               #'clf__kernel': ["linear","rbf","poly"],
                               #'clf__gamma': [0.1,0.001,0.00001],
                               #'clf__C': []}],
                               #'clf__max_features': [0.1,0.2,0.5],
                               #'clf__min_samples_leaf': [1],
                               #'clf__n_estimators': [10,25,50]}],
                               #'clf__alpha': [10, 0.1, 0.001]}],
                  error_score=999,
                  cv=5,
                  n_jobs=1,
                  verbose=10,
                  scoring=make_scorer(hamming_loss))

pipe = OneVsRestClassifier(SVC(kernel="linear", C=0.01))
pipe.fit(X_train, y_train)

best_pipe = pipe
#best_pipe = gs.best_estimator_
#print gs.best_params_

#means = gs.cv_results_['mean_test_score']
#stds = gs.cv_results_['std_test_score']
#for mean, std, params in zip(means, stds, gs.cv_results_['params']):
#    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

best_pipe.fit(X_train, y_train)


y_test_predicted = best_pipe.predict(X_test)
score_test = [hamming_loss(y_test[:,i], y_test_predicted[:,i]) for i in range(0,3)]
print("hamming_loss test data: " + str(score_test))


y_train_predicted = best_pipe.predict(X_train)
score_train = [hamming_loss(y_train[:,i], y_train_predicted[:,i]) for i in range(0,3)]
print("hamming_loss train data: " + str(score_train))


Data_test = []
for i in range(1, 139):
    imagefile = nib.load("data/set_test/test_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    Data_test.append(np.asarray(I))

X_test = Data_test
X_test = ccc.transform(X_test)
X_test = np.array(X_test)
print "Cubes"
print X_test.shape
#
X_test=v.transform(X_test)
print "Variance"
print X_test.shape
#
X_test=sc.transform(X_test)
#
X_test=sel.transform(X_test)
print "Mutual Info"
print X_train.shape
print X_test.shape
predictions = best_pipe.predict(X_test)
gender = ['gender,False' if x == 0 else 'gender,True' for x in predictions]

with open("Scores_gender.csv", mode='w') as f:
    f.write("ID,Prediction\n")
    index=0
    for idx, pred in enumerate(gender):
        f.write(str(index)+','+str(idx)+','+str(pred)+'\n\n\n')
        index = index + 1