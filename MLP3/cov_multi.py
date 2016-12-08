import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

Targets = []
target_file=open('data/targets.csv','r')
for lines in target_file:
    lines = lines.strip('\n')
    lines = lines.split(',')
    #lines = lines[2]
    Targets.append(lines)


Data = []
for i in range(1, 279):
    imagefile = nib.load("data/set_train/train_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:,:,80, 0]
    I=I.flatten(order='C')
    imagefile.uncache()
    Data.append(I)

Data = np.asarray(Data)

Data = np.transpose(Data)
Targets = np.asarray(Targets, dtype=float)

cov_matrix = []

cov_2 = Targets - np.asarray([np.mean(Targets, axis=0),]*278)

for pixel in Data:
    cov_1 = (pixel - np.mean(pixel)*np.ones((1,len(pixel))))
    cov_matrix.append(np.mean(np.dot(cov_1,cov_2)))

cov_matrix = np.asarray(cov_matrix)

plt.imshow(cov_matrix.reshape((176,208)))
plt.show()