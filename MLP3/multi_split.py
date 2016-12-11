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

Targets = []
target_file=open('data/targets.csv','r')
for lines in target_file:
    lines = lines.strip('\n')
    lines = lines.split(',')
    Targets.append(lines)

Targets = np.asarray(Targets)

Data = []
for i in range(1, 279):
    imagefile = nib.load("data/set_train/train_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:,:,:,0]
    imagefile.uncache()
    Data.append(np.asarray(I))

Data = np.asarray(Data)

X_train, X_test, y_train, y_test = \
      train_test_split(Data, Targets, test_size=0.33, random_state=42)#, stratify=Targets)

#X_train = Data
#y_train = Targets

print 'Fitting process started'

forest = RandomForestClassifier(n_estimators=1, random_state=1)

clf = MultiOutputClassifier(forest)

pipe_list=[]

for Class in np.transpose(Targets):
    Class = np.transpose(Class)

    pipe = Pipeline([('cut', CenterCutCubes(size_cubes=5, plane_jump=1, x1=50, y1=80, z1=50, x2=120, y2=150, z2=100)),
                      #('var', VarianceThreshold()),
                      #('sel', Select(type='mutual_info', threshold=0.1)),
                      #('scl', StandardScaler()),
                      #('pca', PCA(n_components=5)),
                      ('clf',clf)])
    print 'pipe done'
    pipe.fit(X_train,y_train)

    print "Train - hamming loss :", hamming_loss(y_train.flatten(),pipe.predict(X_train).flatten())
    print "Test - hamming loss :", hamming_loss(y_test.flatten(),pipe.predict(X_test).flatten())

    pipe_list.append(pipe)

Data_test = []
for i in range(1, 139):
    imagefile = nib.load("data/set_test/test_"+str(i)+".nii")
    image = imagefile.get_data()
    I=image[:,:,:,0]
    imagefile.uncache()
    Data_test.append(np.asarray(I))

output_data = []

for index in range(0,3):
    output_data.append(pipe_list[index].predict(Data_test))

output_data=np.asarray(output_data)
output_data = np.column_stack((output_data[0],output_data[1],output_data[2]))
print output_data.shape


###########################################
#      WRITING DATA IN OUTPUT FORMAT      #
###########################################

#GENDER (gender is not actually binary - progressive thought leads to a progressive society

gender = output_data[:,0]
age = output_data[:,1]
health = output_data[:,2]

gender = ['gender,False' if x == '0' else 'gender,True' for x in gender]
age = ['age,False' if x == '0' else 'age,True' for x in age]
health = ['health,False' if x == '0' else 'health,True' for x in health]

output_labeled = np.column_stack((gender,age,health))

with open("sub_base.csv", mode='w') as f:
    f.write("ID,Sample,Label,Predicted\n")
    index=0
    for idx, pred in enumerate(output_labeled):
        for i in pred:
            f.write(str(index)+','+str(idx)+','+str(i)+'\n')
            index = index + 1

print "Test image successfully processed"