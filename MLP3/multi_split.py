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
from sklearn.metrics import hamming_loss
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier\

from scipy import ndimage
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from skimage.filters import gaussian

import math

Targets = []
Gender = []
Age = []
Health = []

x1=50
y1=80
z1=50
x2=120
y2=150
z2=100

target_file=open('data/targets.csv','r')
for lines in target_file:
    lines = lines.strip('\n')
    lines = lines.split(',')
    Gender.append(lines[0])
    Age.append(lines[1])
    Health.append(lines[2])
    Targets.append(lines)

Targets = np.asarray(Targets, dtype=int)

Data = []
for i in range(1, 279):
    imagefile = nib.load("data/set_train/train_"+str(i)+".nii")
    image = imagefile.get_data()
    image_dim = np.asarray(I).shape
    I=I.flatten(order='C')
    I = np.trim_zeros(I)
    imagefile.uncache()
    Data.append(np.asarray(I))

Data = np.asarray(Data)

####################
#    TESTING DATA  #
####################

Data_test = []
for i in range(1, 139):
    imagefile = nib.load("data/set_test/test_"+str(i)+".nii")
    image = imagefile.get_data()
    I=I.flatten(order='C')
    I = np.trim_zeros(I)
    imagefile.uncache()
    Data_test.append(np.asarray(I))

Data_test=np.asarray(Data_test)

print 'Fitting process started'


######################
#    GENDER COV      #
######################

Data_gender = np.transpose(np.asarray(Data))
Data_test_gender = np.transpose(np.asarray(Data_test))
Targets_gender = np.asarray(Gender, dtype=float)

index = 0
ignore_index = []
cov_matrix = []

cov_2 = Targets_gender - np.asarray([np.mean(Targets_gender, axis=0),]*278)

for pixel in Data_gender:
    #print 'pixel', index+1
    cov_1 = (pixel - np.mean(pixel)*np.ones((1,len(pixel))))
    covarience_value = abs(np.mean(np.dot(cov_1,cov_2)))
    correlation_value = covarience_value/(np.var(pixel)*np.var(Targets_gender))
    if correlation_value < 0.5 or math.isnan(correlation_value) == True:
        ignore_index.append(index)
        cov_matrix.append(0)
    else:
        cov_matrix.append(abs(np.mean(np.dot(cov_1, cov_2))))
    index = index + 1


Data_gender = np.transpose(np.delete(Data_gender, ignore_index, axis=0))
Data_test_gender = np.transpose(np.delete(Data_test_gender, ignore_index, axis=0))
cov_matrix = np.asarray(cov_matrix)

print "processed gender image"

#plt.imshow(cov_matrix.reshape(image_dim))
#plt.show()

######################
#    AGE COV         #
######################

Data_Age = np.transpose(np.asarray(Data))
Data_test_age = np.transpose(np.asarray(Data_test))
Targets_Age = np.asarray(Age, dtype=float)

index = 0
ignore_index = []
cov_matrix = []

cov_2 = Targets_Age - np.asarray([np.mean(Targets_Age, axis=0),]*278)

for pixel in Data_Age:
    #print 'pixel', index+1
    cov_1 = (pixel - np.mean(pixel)*np.ones((1,len(pixel))))
    covarience_value = abs(np.mean(np.dot(cov_1,cov_2)))
    correlation_value = covarience_value/(np.var(pixel)*np.var(Targets_gender))
    if correlation_value < 0.5 or math.isnan(correlation_value) == True:
        ignore_index.append(index)
        cov_matrix.append(0)
    else:
        cov_matrix.append(abs(np.mean(np.dot(cov_1, cov_2))))
    index = index + 1

Data_Age = np.transpose(np.delete(Data_Age, ignore_index, axis=0))
Data_test_age = np.transpose(np.delete(Data_test_age, ignore_index, axis=0))
cov_matrix = np.asarray(cov_matrix)

print "processed age image"

######################
#    Health COV      #
######################

Data_Health = np.transpose(np.asarray(Data))
Data_test_health = np.transpose(np.asarray(Data_test))
Targets_Health = np.asarray(Health, dtype=float)

index = 0
ignore_index = []
cov_matrix = []

cov_2 = Targets_Health - np.asarray([np.mean(Targets_Health, axis=0),]*278)

for pixel in Data_Health:
    #print 'pixel', index+1
    cov_1 = (pixel - np.mean(pixel)*np.ones((1,len(pixel))))
    covarience_value = abs(np.mean(np.dot(cov_1,cov_2)))
    correlation_value = covarience_value/(np.var(pixel)*np.var(Targets_gender))
    if correlation_value < 0.5 or math.isnan(correlation_value) == True:
        ignore_index.append(index)
        cov_matrix.append(0)
    else:
        cov_matrix.append(abs(np.mean(np.dot(cov_1, cov_2))))
    index = index + 1

Data_Health = np.transpose(np.delete(Data_Health, ignore_index, axis=0))
Data_test_health = np.transpose(np.delete(Data_test_health, ignore_index, axis=0))
cov_matrix = np.asarray(cov_matrix)

print "processed health image"

#################################
#         FITTING PROCESS       #
#################################

from sklearn import naive_bayes
from sklearn import tree
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn import ensemble

from sklearn.neural_network import MLPClassifier


#forest = OneVsRestClassifier(RandomForestClassifier(n_estimators=5,random_state=1))
#clf = MultiOutputClassifier(forest)

clf = naive_bayes.GaussianNB()

pipe_list=[]
Target_class=[]

for Class in np.transpose(Targets):
    Target_class.append(np.transpose(Class))

################
#    GENDER    #
################

#X_train, X_test, y_train, y_test = train_test_split(Data_gender, Target_class[0], test_size=0.33, random_state=42, stratify=Target_class[0])

X_train = Data_gender
y_train = Target_class[0]

clf_gender = clf

#svm_model = SVC(C=0.0001, kernel='linear', class_weight={0: 6, 1: 1}, probability=True)

pipe_gender = Pipeline([#('cut', CenterCutCubes(size_cubes=5, plane_jump=1, x1=50, y1=80, z1=50, x2=120, y2=150, z2=100)),
                #('var', VarianceThreshold()),
                #('sel', Select(type='mutual_info', threshold=0.1)),
                ('scl', StandardScaler()),
                ('pca', PCA(n_components=5)),
                ('clf', clf_gender)
                ])

print 'gender pipe done'
pipe_gender.fit(X_train,y_train)

print "Train - hamming loss :", hamming_loss(y_train.flatten(),pipe_gender.predict(X_train).flatten())
#print "Test - hamming loss :", hamming_loss(y_test.flatten(),pipe_gender.predict(X_test).flatten())

output_data_gender = pipe_gender.predict(Data_test_gender)
print "gender prediction was successful"

################
#    AGE       #
################

#X_train, X_test, y_train, y_test = train_test_split(Data_Age, Target_class[1], test_size=0.33, random_state=42, stratify=Target_class[1])

X_train = Data_Age
y_train = Target_class[1]

clf_age = clf

pipe_age = Pipeline([#('cut', CenterCutCubes(size_cubes=5, plane_jump=1, x1=50, y1=40, z1=70, x2=160, y2=135, z2=120)),
                  ('var', VarianceThreshold()),
                  #('sel', Select(type='mutual_info', threshold=0.1)),
                  ('scl', StandardScaler()),
                  ('pca', PCA(n_components=15)),
                  ('clf',clf_age)])
print 'age pipe done'
pipe_age.fit(X_train,y_train)

print "Train - hamming loss :", hamming_loss(y_train.flatten(),pipe_age.predict(X_train).flatten())
#print "Test - hamming loss :", hamming_loss(y_test.flatten(),pipe_age.predict(X_test).flatten())

output_data_age = pipe_age.predict(Data_test_age)
print "age prediction was successful"

################
#    HEALTH    #
################

#X_train, X_test, y_train, y_test = train_test_split(Data_Health, Target_class[2], test_size=0.33, random_state=42, stratify=Target_class[2])

X_train = Data_Health
y_train = Target_class[2]

clf_health = linear_model.LogisticRegression()

pipe_health = Pipeline([#('cut', CenterCutCubes(size_cubes=5, plane_jump=1, x1=50, y1=40, z1=70, x2=160, y2=135, z2=120)),
                  ('var', VarianceThreshold()),
                  #('sel', Select(type='mutual_info', threshold=0.1)),
                  ('scl', StandardScaler()),
                  ('pca', PCA(n_components=5)),
                  ('clf',clf_health)])
print 'health pipe done'
pipe_health.fit(X_train,y_train)

print "Train - hamming loss :", hamming_loss(y_train.flatten(),pipe_health.predict(X_train).flatten())
#print "Test - hamming loss :", hamming_loss(y_test.flatten(),pipe_health.predict(X_test).flatten())

output_data_health = pipe_health.predict(Data_test_health)
print "health prediction was successful"

output_data = np.column_stack((output_data_gender,output_data_age,output_data_health))
print output_data.shape

###########################################
#      WRITING DATA IN OUTPUT FORMAT      #
###########################################

#GENDER (gender is not actually binary - progressive thought leads to a progressive society)

output_data_gender = ['gender,False' if x == 0 else 'gender,True' for x in output_data_gender]
output_data_age = ['age,False' if x == 0 else 'age,True' for x in output_data_age]
output_data_health = ['health,False' if x == 0 else 'health,True' for x in output_data_health]

output_labeled = np.column_stack((output_data_gender,output_data_age,output_data_health))

with open("sub_alt.csv", mode='w') as f:
    f.write("ID,Sample,Label,Predicted\n")
    index=0
    for idx, pred in enumerate(output_labeled):
        for i in pred:
            f.write(str(index)+','+str(idx)+','+str(i)+'\n')
            index = index + 1

print "Test image successfully processed"