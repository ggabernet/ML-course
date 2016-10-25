from skimage import measure
import numpy as np
import nibabel as nib
from feature_extraction import Intensities, CenterCut, Covariance, Filtering, Contours
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
	Data.append(np.asarray(I))

print I.shape

X_train = Data
y_train = Targets

# X_train, X_test, y_train, y_test = \
#     train_test_split(Data, Targets, test_size=0.33, random_state=42)

size_cubes = 5
iterations = 8
cubes = Intensities()
cubes.calculate_intensity_cubes(X_train, size_cubes=size_cubes, iterations=iterations)
desc_train = cubes.descriptor

# cut.make_cut(X_test)
# cut_test = cut.cut
#
# filter.calculate_prewitt(cut_test)
# desc_test = filter.flatten(filter.transformed)

print desc_train.shape

pipe = Pipeline([('scl', StandardScaler()),
				 ('var', VarianceThreshold()),
				 ('pca', PCA(n_components=100)),
				 ('clf', SVR(kernel='linear'))])
param_range_svm = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100]
param_range_lasso = np.linspace(0, 10, 11)
gs = GridSearchCV(estimator=pipe,
				  param_grid=[{'clf__C': [1],
							   'pca__n_components': [200]}],
				  cv=5,
				  scoring=make_scorer(mean_squared_error),
				  n_jobs=1)

gs.fit(desc_train, y_train)

best_pipe = gs.best_estimator_
print gs.best_params_

means = gs.cv_results_['mean_test_score']
stds = gs.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, gs.cv_results_['params']):
	if params == gs.best_params_:
		print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

best_pipe.fit(desc_train, y_train)

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


Data_test = []
for i in range(1, 139):
	imagefile = nib.load("data/set_test/test_"+str(i)+".nii")
	image = imagefile.get_data()
	I = image[:, :, :, 0]
	Data_test.append(np.asarray(I))


cubes.calculate_intensity_cubes(Data_test,size_cubes=size_cubes, iterations=iterations)
desc_real_test = cubes.descriptor

predictions = best_pipe.predict(desc_real_test)

with open("SubmissionOptimizedCombinedModel.csv", mode='w') as f:
	f.write("ID,Prediction\n")
	for idx, pred in enumerate(predictions):
		f.write(str(idx+1)+','+str(pred)+'\n')