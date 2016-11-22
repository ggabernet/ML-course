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

from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

Targets = np.genfromtxt("data/targets.csv")

Data = []
for i in range(1, 279):
    print "Train image ", str(i), "processed"
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

clf = linear_model.LogisticRegression()
print "Method : SVC, C=0.1, kernel=linear, n_pca=250"

pipe = Pipeline([('cut', CenterCutCubes(size_cubes=4, plane_jump=1, x1=50, x2=120, y1=50, y2=150, z1=50, z2=100)),
                ('var', VarianceThreshold()),
                ('sel', Select(type='mutual_info', threshold=0.5)),
                ('scl', StandardScaler()),
                ('pca', PCA(n_components=10)),
                ('clf', clf)])

pipe.fit(X_train, y_train)
#X_train_t = pipe.transform(X_train)
#print X_train_t.shape




print "log loss train :", log_loss(y_train, pipe.predict_proba(X_train))
print "log loss test :", log_loss(y_test, pipe.predict_proba(X_test))


##############################
#      Submission.csv        #
##############################

x_low=50
x_up=120
y_low=50
y_up=150
z_low=50
z_up=100

Data_test = []
for i in range(1, 139):
    print "Test image ", str(i), "processed"
    imagefile = nib.load("data/set_test/test_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    I = I[x_low:x_up, y_low:y_up, z_low:z_up]
    I=I.flatten(order='C')
    Data_test.append(np.asarray(I))

X_test = Data_test

predictions = pipe.predict_proba(X_test)

with open("Submission_cube.csv", mode='w') as f:
    f.write("ID,Prediction\n")
    for idx, pred in enumerate(predictions):
        f.write(str(idx+1)+','+str(pred[1])+'\n')

# svm_model = SVC(C=0.1, kernel='rbf', gamma=0.00001, class_weight={0: 3, 1: 1}, probability=True)
#
#
# gs = GridSearchCV(estimator=svm_model,
#                   param_grid={'C': [1, 0.1, 0.01],
#                               'gamma': [1, 0.1, 0.001, 0.00001],
#                               'kernel': ['linear', 'rbf']},
#                   error_score=999,
#                   cv=5,
#                   n_jobs=-1,
#                   verbose=10,
#                   scoring=make_scorer(log_loss))
#
# gs.fit(X_train_t, y_train)
#
# means = gs.cv_results_['mean_test_score']
# stds = gs.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, gs.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
#
# best_estim = gs.best_estimator_
# best_estim.fit(X_train_t, y_train)
# print gs.best_params_
#
#
# X_test_t = pipe.transform(X_test)
#
# y_test_predicted = best_estim.predict_proba(X_test_t)
# score_test = log_loss(y_test, y_test_predicted)
# print("log_loss test data: " + str(score_test))
#
#
# y_train_predicted = best_estim.predict_proba(X_train_t)
# score_train = log_loss(y_train, y_train_predicted)
# print("log_loss train data: " + str(score_train))
# scoreMCC_train = matthews_corrcoef(y_train, y_train_predicted)
# print("MCC score train data: " + str(scoreMCC_train))
#
#
# # Training model again with all the training data (without split)
# X_train = Data
# y_train = Targets
#
# X_train_t = pipe.transform(X_train)
# best_estim.fit(X_train_t, y_train)
#
# # Generating score file for Kaggle test data
# Data_test = []
# for i in range(1, 139):
#     imagefile = nib.load("data/set_test/test_"+str(i)+".nii")
#     image = imagefile.get_data()
#     I = image[:, :, :, 0]
#     Data_test.append(np.asarray(I))
#
# X_test = Data_test
# X_test_t = pipe.transform(X_test)
#
# predictions = best_estim.predict_proba(X_test_t)
#
# with open("Scoresbagging.csv", mode='w') as f:
#     f.write("ID,Prediction\n")
#     for idx, pred in enumerate(predictions):
#         f.write(str(idx+1)+','+str(pred[1])+'\n')
