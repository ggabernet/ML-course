from skimage import measure
import numpy as np
import nibabel as nib
from feature_extraction import Intensities
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import *
from sklearn.linear_model import Lasso
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR
from sklearn.decomposition import PCA, RandomizedPCA

Targets = np.genfromtxt("data/targets.csv")

Data = []
for i in range(1, 279):
    imagefile = nib.load("data/set_train/train_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    Data.append(np.asarray(I))

print I.shape


X_train, X_test, y_train, y_test = \
    train_test_split(Data, Targets, test_size=0.33, random_state=42)

cnt = Intensities()
cnt.calculate_intensity_cubes(X_train, size_cubes=3, iterations=3)
desc_train = cnt.descriptor
cnt.calculate_intensity_cubes(X_test, size_cubes=3, iterations=3)
desc_test = cnt.descriptor

print desc_train.shape

pipe = Pipeline([('var', VarianceThreshold(threshold=0)),
                 ('pca', RandomizedPCA(n_components=10)),
                 ('clf', SVR())])
param_range_svm = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100]
param_range_lasso = np.linspace(0, 10, 11)
gs = GridSearchCV(estimator=pipe,
                  param_grid=[{'var__threshold': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                               'clf__C': [0.001, 0.01, 0.1, 1.0],
                               'pca__n_components': np.linspace(10,100,33, dtype=int)}],
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
for i in range(1, 139):
    imagefile = nib.load("data/set_test/test_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    Data_test.append(I)

cnt.calculate_intensity_cubes(Data_test, size_cubes=3, iterations=5)
X_real_test = cnt.descriptor
predictions = best_pipe.predict(X_real_test)

with open("SubmissionIntensitiesCubes.csv", mode='w') as f:
    f.write("ID,Prediction\n")
    for idx, pred in enumerate(predictions):
        f.write(str(idx+1)+','+str(pred)+'\n')
