import nibabel as nib
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV

Targets = np.genfromtxt("data/targets.csv")

Data_df = pd.read_csv("All_images_central_area_corr2000.csv", header=0, index_col=0)

Data = np.asarray(Data_df).transpose()

X_train, X_test, y_train, y_test = \
    train_test_split(Data, Targets, test_size=0.33, random_state=42)

# Pipeline that scales (StandardScaler()), performs dimensionality reduction with PCA and trains a support vector
# regression machine classifier.
# pipe_svr = Pipeline([('scl', StandardScaler()),
# 						('pca', PCA(n_components=100)),
# 						('clf', SVR(kernel='linear', C=1))])
# pipe_svr.fit(X_train, y_train)
# print('R^2 score: %.3f' % pipe_svr.score(X_test, y_test))

svr = SVR(C=1.0)
lin = Lasso(alpha=1.0, random_state=42)
gs = GridSearchCV(estimator=lin,
				  param_grid= [{'alpha': np.linspace(0,1000,50,dtype=int)}],
				  cv=10,
				  n_jobs=-1)

gs.fit(X_train, y_train)
best_lin = gs.best_estimator_
print gs.best_params_
print gs.best_score_

print "MRSE test data: " + str(mean_squared_error(y_test,gs.predict(X_test)))
print "MRSE train data: " + str(mean_squared_error(y_train, gs.predict(X_train)))





