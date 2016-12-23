from skimage import measure
import numpy as np
import math
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
from sklearn.metrics import hamming_loss
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.ensemble import *
from sklearn.multioutput import MultiOutputClassifier

from scipy import ndimage
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
import matplotlib.pyplot as plt

Targets = []

target_file=open('data/targets.csv','r')
for lines in target_file:
    lines = lines.strip('\n')
    lines = lines.split(',')
    Targets.append(lines)

Targets = np.asarray(Targets,dtype=int)

x1=50
y1=80
z1=50
x2=120
y2=150
z2=100

Data = []
for i in range(1, 279):
    imagefile = nib.load("data/set_train/train_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:,:,80,0]
    I = np.asarray(I, dtype=float)
    I = I/np.max(I)
    I = I.flatten(order='C')
    imagefile.uncache()
    Data.append(np.asarray(I))

Data = np.asarray(Data)

Data_test = []
for i in range(1, 139):
    imagefile = nib.load("data/set_test/test_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:,:,80,0]
    I = np.asarray(I, dtype=float)
    I = I / np.max(I)
    I = I.flatten(order='C')
    imagefile.uncache()
    Data_test.append(np.asarray(I))

Data_test = np.asarray(Data_test)

#######################
#         COV         #
#######################
#
Data = np.transpose(np.asarray(Data))
Data_test = np.transpose(np.asarray(Data_test))
Targets = np.asarray(Targets, dtype=float)

index = 0
ignore_index = []
cov_matrix = []

cov_2 = Targets - np.asarray([np.mean(Targets, axis=0),]*278)

for pixel in Data:
    #print 'pixel', index+1
    cov_1 = (pixel - np.mean(pixel)*np.ones((1,len(pixel))))
    covarience_value = abs(np.mean(np.dot(cov_1,cov_2)))
    correlation_value = covarience_value/(np.var(pixel)*np.var(Targets))
    if correlation_value < 500 or math.isnan(correlation_value) == True:
        ignore_index.append(index)
        cov_matrix.append(0)
    else:
        cov_matrix.append(abs(np.mean(np.dot(cov_1, cov_2))))
    index = index + 1

Data = np.transpose(np.delete(Data, ignore_index, axis=0))
Data_test = np.transpose(np.delete(Data_test, ignore_index, axis=0))
cov_matrix = np.asarray(cov_matrix)

#plt.imshow(cov_matrix.reshape(image_dim))
#plt.show()

print "processed images"

X_train, X_test, y_train, y_test = \
      train_test_split(Data, Targets, test_size=0.33, random_state=42)#, stratify=Targets)

#X_train = Data
#y_train = Targets

print 'Fitting process started'

forest = OneVsOneClassifier(SVC(kernel='linear',C=5))
clf = MultiOutputClassifier(forest)

pipe = Pipeline([#('cut', CenterCutCubes(size_cubes=5, plane_jump=1, x1=50, y1=80, z1=50, x2=120, y2=150, z2=100)),
                #('var', VarianceThreshold()),
                #('sel', Select(type='mutual_info', threshold=0.1)),
                #('scl', StandardScaler()),
                ('pca', PCA(n_components=250)),
                ('clf', clf)
                ])
print 'pipe done'

pipe.fit(X_train,y_train)

print "Train - hamming loss :", hamming_loss(y_train,pipe.predict(X_train))
print "Test - hamming loss :", hamming_loss(y_test,pipe.predict(X_test))

output_data = pipe.predict(Data_test)

###########################################
#      WRITING DATA IN OUTPUT FORMAT      #
###########################################

#GENDER (gender is not actually binary - progressive thought leads to a progressive society

gender = output_data[:,0]
age = output_data[:,1]
health = output_data[:,2]

gender = ['gender,False' if x == 0 else 'gender,True' for x in gender]
age = ['age,False' if x == 0 else 'age,True' for x in age]
health = ['health,False' if x == 0 else 'health,True' for x in health]

output_labeled = np.column_stack((gender,age,health))

with open("sub_ALEX.csv", mode='w') as f:
    f.write("ID,Sample,Label,Predicted\n")
    index=0
    for idx, pred in enumerate(output_labeled):
        for i in pred:
            f.write(str(index)+','+str(idx)+','+str(i)+'\n')
            index = index + 1

print "Test image successfully processed"