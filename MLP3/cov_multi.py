import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import math

Gender = []
Age = []
Health = []
Target_all = []

target_file=open('data/targets.csv','r')
for lines in target_file:
    lines = lines.strip('\n')
    lines = lines.split(',')
    Gender.append(lines[0])
    Age.append(lines[1])
    Health.append(lines[2])
    Target_all.append(lines)

Data = []
for i in range(1, 279):
    imagefile = nib.load("data/set_train/train_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:,:,80, 0]
    I=I.flatten(order='C')
    I = np.trim_zeros(I)
    imagefile.uncache()
    Data.append(I)

Data = np.asarray(Data)
Data = np.transpose(Data)

Targets = np.asarray(Target_all, dtype=float)

cov_matrix = []

cov_2 = Targets - np.asarray([np.mean(Targets, axis=0),]*278)

for pixel in Data:
    cov_1 = (pixel - np.mean(pixel)*np.ones((1,len(pixel))))
    covarience_value = abs(np.mean(np.dot(cov_1, cov_2)))
    correlation_value = covarience_value / (np.var(pixel) * np.var(Targets))
    if correlation_value < 0.25 or math.isnan(correlation_value) == True:
        cov_matrix.append(0)
    else:
        cov_matrix.append(correlation_value)

cov_matrix = np.asarray(cov_matrix)

print np.max(cov_matrix)

plt.imshow(cov_matrix.reshape(image_dim))
plt.show()

Gender = np.asarray(Gender, dtype=float)
Age = np.asarray(Age, dtype=float)
Health = np.asarray(Health, dtype=float)

print np.count_nonzero(Gender)
print np.count_nonzero(Age)
print np.count_nonzero(Health)

print "****GENDER****"

print 'Gender and Gender :', np.corrcoef(Gender,Gender)[0][1]
print 'Gender and Age :', np.corrcoef(Gender,Age)[0][1]
print 'Gender and Health :', np.corrcoef(Gender,Health)[0][1]

print '****AGE****'
print 'Age and Gender :', np.corrcoef(Age,Gender)[0][1]
print 'Age and Age :', np.corrcoef(Age,Age)[0][1]
print 'Age and Health :', np.corrcoef(Age,Health)[0][1]

print '****HEALTH****'
print 'Health and Gender :', np.corrcoef(Health,Gender)[0][1]
print 'Health and Age :', np.corrcoef(Health,Age)[0][1]
print 'Health and Health :', np.corrcoef(Health,Health)[0][1]