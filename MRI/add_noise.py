import random 
from skimage import io
import numpy 

image=io.imread('image.jpg')
#image should be greyscale
image=np.asarray(image, dtype='float')
image_norm=image/np.max(image)

[dim_1,dim_2]=image_norm.shape

random_matrix=np.random.rand(dim_1,dim_2)
random_noise=random_matrix*0.5 #50% noise contribution 
image_with_noise=(image_norm+random_noise)/np.max(image_norm+random_noise)
