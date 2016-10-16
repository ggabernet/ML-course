import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.decomposition import PCA
from skimage.filters import threshold_otsu
import skimage.filters
from numpy.linalg import inv, det

target_list=[]

targets=open('data/targets.csv','r')
for lines in targets:
    target_list.append(int(lines.strip('\n')))

os.chdir('data/set_train')
L=os.listdir('./')

target_min=np.amin(target_list)
target_max=np.amax(target_list)

increment=10

bins=range(target_min,target_max,increment)
target_bin = np.digitize(target_list, bins)
target_bin = target_bin.tolist()

data_total=[]

for j in list(set(target_bin)):
    indices=[]
    index=0
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
        img=nib.load(L[i])
        data=img.get_data()
        data=data[:,:,80,0]
        data=np.asarray(data)
        scans.append(data)

        data_total.append(data)

#mean_scan=np.average(np.array(scans),weights=target_list,axis=0)
    std_scans=np.std(np.array(scans),axis=0)
    mean_scans = np.mean(np.array(scans), axis=0)

    mean_scans=np.asarray(mean_scans)
    scans=np.asarray(scans)

    A=[]

    for i in range(number_of_scans):

        scans_transpose=np.transpose(scans[i])

        A.append(np.dot(np.dot(mean_scans,scans_transpose),inv(np.dot(scans[i],scans_transpose))))

        A_transform = np.sum(A)/number_of_scans

    plt.imshow(std_scans)
    # plt.plot(target_list)
    plt.show()

    print "Average standard deviation :",np.average(std_scans)
    print ""

print "-------------------"
print ""

std_total=np.std(np.array(data_total),axis=0)
print "Average standard deviation between all scans :",np.average(std_total)
print ""

print "-------------------"
print ""




########-PCA dimension reduction-#######

#pca = PCA(n_components=10)
#pca.fit(data)

#print pca.explained_variance_ratio_
#print pca.components_

########################################



############-distrubtion of scans-#############

#plt.hist(target_list,20)
#plt.title("Number of scans from different age groups")
#plt.show()

###############################################