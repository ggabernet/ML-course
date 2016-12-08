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
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt

Targets = np.genfromtxt("data/targets.csv")

Data = []
for i in range(1, 279):
    imagefile = nib.load("data/set_train/train_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    imagefile.uncache()
    Data.append(np.asarray(I))

print I.shape

Data = np.asarray(Data)
print Data.shape

X_train, X_test, y_train, y_test = \
      train_test_split(Data, Targets, test_size=0.33, random_state=42, stratify=Targets)

print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape
print y_test

pipe = Pipeline([('cut', CenterCutCubes(size_cubes=2, plane_jump=1)),
                ('var', VarianceThreshold(0.01)),
                ('sel', Select(type='mutual_info', threshold=0.1)),
                ('scl', StandardScaler()),
                ('pca', PCA(n_components=150))])


pipe.fit(X_train, y_train)
X_train_t = pipe.transform(X_train)
print X_train_t.shape

bagging = BaggingClassifier(base_estimator=SVC(C=0.00001, kernel='linear', class_weight={0: 6, 1: 1}, probability=True),
                            n_estimators=10,
                            max_samples=0.5,
                            max_features=1.0,
                            bootstrap=True,
                            n_jobs=-1,
                            random_state=42,
                            verbose=1)


bagging.fit(X_train_t, y_train)


X_test_t = pipe.transform(X_test)

y_test_predicted = bagging.predict_proba(X_test_t)
score_test = log_loss(y_test, y_test_predicted)
print("log_loss test data: " + str(score_test))


y_train_predicted = bagging.predict_proba(X_train_t)
score_train = log_loss(y_train, y_train_predicted)
print("log_loss train data: " + str(score_train))


# Training model again with all the training data (without split)
X_train = Data
y_train = Targets

X_train_t = pipe.transform(X_train)
bagging.fit(X_train_t, y_train)

# Generating score file for Kaggle test data
Data_test = []
for i in range(1, 139):
    imagefile = nib.load("data/set_test/test_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    Data_test.append(np.asarray(I))

X_test = Data_test
X_test_t = pipe.transform(X_test)

predictions = bagging.predict_proba(X_test_t)

with open("Scores_bagging.csv", mode='w') as f:
    f.write("ID,Prediction\n")
    for idx, pred in enumerate(predictions):
        f.write(str(idx+1)+','+str(pred[1])+'\n')
