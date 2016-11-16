from skimage import measure
import numpy as np
import nibabel as nib
from feature_extraction import CenterCutCubes, CovSel, CenterCut
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import *
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.decomposition import PCA, RandomizedPCA

Targets = np.genfromtxt("data/targets.csv")

Data = []
for i in range(1, 279):
    imagefile = nib.load("data/set_train/train_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    imagefile.uncache()
    Data.append(np.asarray(I))

print I.shape

X_train = Data
y_train = Targets

# X_train, X_test, y_train, y_test = \
#     train_test_split(Data, Targets, test_size=0.33, random_state=42)

# cut.make_cut(X_test)
# cut_test = cut.cut
#
# filter.calculate_prewitt(cut_test)
# desc_test = filter.flatten(filter.transformed)

cut = CenterCutCubes(size_cubes=5)
cut.fit(X_train)
X_cut = cut.transform(X_train)

print X_cut.shape

std = StandardScaler()
std.fit(X_cut)
X_std = std.transform(X_cut)

print X_std.shape

var = VarianceThreshold()
var.fit(X_std)
X_var = var.transform(X_std)

print X_var.shape

pca = PCA(n_components=250)
pca.fit(X_var)
X_pca = pca.transform(X_var)

print X_pca.shape

pipe = Pipeline([('clf', SVR(kernel='linear'))])
param_range_svm = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100]
param_range_lasso = np.linspace(0, 10, 11)
param_range_cut_left = range(20, 70, 10)
param_range_cut_right = range(80, 160, 10)
param_range_size_cube = range(1, 10)
gs = GridSearchCV(estimator=pipe,
                  param_grid=[{'clf__C': [1]}],
                  error_score=999,
                  cv=5,
                  n_jobs=1,
                  scoring=make_scorer(mean_squared_error))

gs.fit(X_pca, y_train)

best_pipe = gs.best_estimator_
print gs.best_params_

means = gs.cv_results_['mean_test_score']
stds = gs.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

#best_pipe.fit(X_train, y_train)

# score = best_pipe.score(desc_test, y_test)
# print("Score R^2: "+str(score))
#
# y_test_predicted = best_pipe.predict(desc_test)
# MRSE_test = mean_squared_error(y_test, y_test_predicted)
# print("MRSE score test data: " + str(MRSE_test))
#
# y_train_predicted = best_pipe.predict(desc_train)
# MRSE_train = mean_squared_error(y_train, y_train_predicted)
# print("MRSE score train data: " + str(MRSE_train))


# Data_test = []
# for i in range(1, 139):
#     imagefile = nib.load("data/set_test/test_"+str(i)+".nii")
#     image = imagefile.get_data()
#     I = image[:, :, :, 0]
#     Data_test.append(np.asarray(I))
#
# X_test = Data_test
#
# predictions = best_pipe.predict(X_test)

# with open("SubmissionOptimizedCombinedModel_2510.csv", mode='w') as f:
#     f.write("ID,Prediction\n")
#     for idx, pred in enumerate(predictions):
#         f.write(str(idx+1)+','+str(pred)+'\n')
