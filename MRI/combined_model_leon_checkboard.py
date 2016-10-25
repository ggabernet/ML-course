from skimage import measure
import numpy as np
import nibabel as nib
from feature_extraction_leon import Intensities, CenterCut, CheckrPixl, Covariance, Filtering, Contours
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import *
from sklearn.linear_model import LassoCV
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.decomposition import PCA, RandomizedPCA, KernelPCA
from itertools import chain
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV

import matplotlib
from matplotlib import pyplot as plt

Targets = np.genfromtxt("data/targets.csv")

Data = []
for i in range(1, 279):
    imagefile = nib.load("data/set_train/train_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    Data.append(np.asarray(I))

print I.shape


#X_train, X_test, y_train, y_test = \
 #   train_test_split(Data, Targets, test_size=0.33, random_state=42)
X_train,y_train=Data,Targets

cut = CenterCut()
cut.make_cut(X_train)
cut_train = cut.cut
cut.make_cubes(cut.cut, size_cubes=5)
desc_train = cut.descriptor
desc_train = desc_train.round(0)
#check = CheckrPixl()
#checker = check.make_checker(cut_train)
#desc_train = checker.checker

#filter = Filtering()
#filter.calculate_prewitt(cut_train)
#desc_train2 = filter.flatten(filter.transformed)

#desc_train = np.array([list(chain.from_iterable(x)) for x in zip(desc_train.tolist(),desc_train2.tolist())])

#cut.make_cut(X_test)
#cut_test = cut.cut
#cut.make_cubes(cut.cut, size_cubes=5)
#desc_test = cut.descriptor

#filter.calculate_prewitt(cut_test)
#desc_test2 = filter.flatten(filter.transformed)

#checker = check.make_checker(cut_test)
#desc_test = checker.checker

#desc_test = np.array([list(chain.from_iterable(x)) for x in zip(desc_test.tolist(),desc_test2.tolist())])

print desc_train.shape
#print desc_test.shape


#('feature_selection', SelectFromModel(LassoCV(),threshold=0.001)),
pipe = Pipeline([('scl', MinMaxScaler(feature_range=(0,1))),
                 ('var', VarianceThreshold()),
                 ('pca', PCA(n_components=100)),
                 ('clf', SVR(kernel='linear', C=1))])

param_range_svm = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100]
param_range_lasso = np.linspace(0, 10, 11)
gs = GridSearchCV(estimator=pipe,
                  param_grid=[{'clf__C': [1], 'pca__n_components': [250]}],
                  cv=10,
                  n_jobs=1)

gs.fit(desc_train, y_train)

best_pipe = gs.best_estimator_
print gs.best_params_
best_pipe.fit(desc_train, y_train)


#score = best_pipe.score(desc_test, y_test)
#print("Score R^2: "+str(score))

#y_test_predicted = best_pipe.predict(desc_test)
#MRSE_test = mean_squared_error(y_test, y_test_predicted)
#print("MRSE score test data: " + str(MRSE_test))

y_train_predicted = best_pipe.predict(desc_train)
MRSE_train = mean_squared_error(y_train, y_train_predicted)
print("MRSE score train data: " + str(MRSE_train))


Data_test = []
for i in range(1, 139):
    imagefile = nib.load("data/set_test/test_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    Data_test.append(np.asarray(I))


cut.make_cut(Data_test)
cut_real_test = cut.cut
cut.make_cubes(cut.cut, size_cubes=5)
desc_real_test = cut.descriptor

#checker = check.make_checker(cut_real_test)
#desc_real_test = checker.checker

#filter.calculate_prewitt(cut_real_test)
#desc_real_test2 = filter.flatten(filter.transformed)

#desc_real_test = np.array([list(chain.from_iterable(x)) for x in zip(desc_real_test.tolist(),desc_real_test2.tolist())])

predictions = best_pipe.predict(desc_real_test)

with open("SubmissionIntensitiesCubicles.csv", mode='w') as f:
    f.write("ID,Prediction\n")
    for idx, pred in enumerate(predictions):
        f.write(str(idx+1)+','+str(pred)+'\n')