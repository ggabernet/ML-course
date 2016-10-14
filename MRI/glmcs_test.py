import matplotlib.pyplot as plt

from skimage.feature import greycomatrix, greycoprops
from skimage import data


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn import preprocessing


img = mpimg.imread('image.jpg')


import nibabel as nib

from scipy.sparse import *
from scipy import *

# open the camera image
#image = data.camera()

import os
import nibabel as nib

os.chdir('data/set_train')
L=os.listdir('./')

I = nib.load(L[0])
data = I.get_data()
data = data[:, :, 80, 0]

data=np.round(data,decimals=0)


min_max_scaler = preprocessing.MinMaxScaler()
scaled_data = min_max_scaler.fit_transform(data)

scaled_data=np.round(scaled_data,decimals=1)

result = greycomatrix(scaled_data, [10], [0], levels=50)

result=result[:,:,0,0]

#plt.contour(data)

#plt.show()

from skimage import measure

contours = measure.find_contours(scaled_data, 0.8)

# Find contours at a constant value of 0.8
contours = measure.find_contours(scaled_data, 0.8)

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(scaled_data, interpolation='nearest', cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()