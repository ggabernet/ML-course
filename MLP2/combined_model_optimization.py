from skimage import measure
import numpy as np
import nibabel as nib
from feature_extraction import CenterCutCubes
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA

Targets = np.genfromtxt("data/targets.csv")

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

pipe = Pipeline([('cut', CenterCutCubes(size_cubes=3)),
                ('scl', StandardScaler()),
                ('var', VarianceThreshold()),
                ('pca', PCA()),
                ('clf', SVC(kernel='linear'))])


gs = GridSearchCV(estimator=pipe,
                  param_grid=[{'cut__size_cubes': [3],
                               'cut__y1': [50],
                               'cut__x1': [50],
                               'cut__z1': [50],
                               'cut__x2': [170],
                               'cut__y2': [200],
                               'cut__z2': [170],
                               'clf__C': [0.1],
                               'pca__n_components': [10]}],
                  error_score=999,
                  cv=5,
                  n_jobs=-1,
                  verbose=10,
                  scoring=make_scorer(log_loss))

gs.fit(X_train, y_train)

best_pipe = gs.best_estimator_
print gs.best_params_

means = gs.cv_results_['mean_test_score']
stds = gs.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

best_pipe.fit(X_train, y_train)

desc_test = best_pipe.transform(X_test)
score = best_pipe.score(desc_test, y_test)
print("Score log_loss: "+str(score))

y_test_predicted = best_pipe.predict(desc_test)
score_test = mean_squared_error(y_test, y_test_predicted)
print("log_loss test data: " + str(score_test))

desc_train = best_pipe.steps[]transform(X_train)
y_train_predicted = best_pipe.predict(desc_train)
score_train = log_loss(y_train, y_train_predicted)
print("log_loss train data: " + str(score_train))


Data_test = []
for i in range(1, 139):
    imagefile = nib.load("data/set_test/test_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    Data_test.append(np.asarray(I))

X_test = Data_test

predictions = best_pipe.predict(X_test)

with open("Scores.csv", mode='w') as f:
    f.write("ID,Prediction\n")
    for idx, pred in enumerate(predictions):
        pred = round(pred, 0)
        f.write(str(idx+1)+','+str(pred)+'\n')
