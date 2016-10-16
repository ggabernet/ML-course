import nibabel as nib
from nibabel import processing as pr
import numpy as np
from sklearn.cross_validation import train_test_split
from feature_extraction import Contours
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import *
from sklearn.linear_model import Lasso
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR

Targets = np.genfromtxt("data/targets.csv")

Data = []
for i in range(1, 279):
    imagefile = nib.load("data/set_train/train_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    Data.append(I)

X_train, X_test, y_train, y_test = \
    train_test_split(Data, Targets, test_size=0.33, random_state=42)

cnt = Contours(intensity=900, min_size=50, layers_x_dim=10)
cnt.calculate_descriptor(X_train)
desc_train = cnt.descriptor
cnt.calculate_descriptor(X_test)
desc_test = cnt.descriptor

print desc_train.shape

pipe = Pipeline([('scl', StandardScaler()),
                 ('var', VarianceThreshold(threshold=0)),
                 ('clf', Lasso())])
param_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
gs = GridSearchCV(estimator=pipe,
                  param_grid=[{'clf__alpha': [1.4]}],
                  cv=5,
                  n_jobs=-1)

gs.fit(desc_train, y_train)

best_pipe = gs.best_estimator_
print gs.best_params_
best_pipe.fit(desc_train, y_train)

score = best_pipe.score(desc_test, y_test)
print("Score R^2: "+str(score))

y_test_predicted = best_pipe.predict(desc_test)
MRSE_test = mean_squared_error(y_test, y_test_predicted)
print("MRSE score test data: " + str(MRSE_test))

y_train_predicted = best_pipe.predict(desc_train)
MRSE_train = mean_squared_error(y_train, y_train_predicted)
print("MRSE score train data: " + str(MRSE_train))


Data_test = []
for i in range(1, 138):
    imagefile = nib.load("data/set_test/test_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    Data_test.append(I)

X_real_test = cnt.calculate_descriptor(Data_test)

with open("Submission.csv",mode='w') as f:
    for idx, pred in enumerate(best_pipe.predict(X_real_test)):
        f.write("ID,Prediction")
        f.write(str(idx)+','+str(pred))
