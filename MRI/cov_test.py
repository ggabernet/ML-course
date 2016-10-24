import nibabel as nib
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.decomposition import PCA
import skimage
from skimage.filters import roberts, sobel, scharr, prewitt
from sklearn.model_selection import GridSearchCV

from skimage import feature
from skimage import exposure
from nibabel import processing

target_list=[]

targets=open('data/targets.csv','r')
for lines in targets:
    target_list.append(int(lines.strip('\n')))

target_min=np.amin(target_list)
target_max=np.amax(target_list)

increment=10

bins=range(target_min,target_max,increment)
target_bin = np.digitize(target_list, bins)
target_bin = target_bin.tolist()

data_total=[]

for j in list(set(target_bin)):
    indices=[]
    index=1
    for entry in target_bin:
        if entry == j:
           indices.append(index)
        index=index+1

    number_of_scans = len(indices)
    if j != np.amax(target_bin):
        print "The age group is :",bins[j-1], "-", bins[j], " The number of samples are :", number_of_scans
    else:
        print "The age group is :", bins[j-1],"-", np.amax(bins)+increment, " The number of samples are :", number_of_scans


    scans=[]

#indices = range(278)

    for i in indices:
        img=nib.load('data/set_train/'+'train_'+str(i)+'.nii')
        data=img.get_data()
        data=data[:,:,80,0]
        data=np.asarray(data)
        scans.append(data)

        data_total.append(data)

#mean_scan=np.average(np.array(scans),weights=target_list,axis=0)
    std_scans=np.std(np.array(scans),axis=0)

    plt.imshow(std_scans)
    plt.show()