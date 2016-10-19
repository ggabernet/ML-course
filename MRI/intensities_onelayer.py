from skimage import measure
import numpy as np
import nibabel as nib

Targets = np.genfromtxt("data/targets.csv")

Data = []
for i in range(1, 2):
    imagefile = nib.load("data/set_train/train_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    Data.append(I)

print I.shape

a, b, c = 9, 9, 3
aaa, bbb = 3, 3
arr = np.arange(a*b*c).reshape(a, b, c)

print arr.shape

arr_view = arr.reshape(a//aaa, aaa, b//bbb, bbb, c)
print arr_view.shape

arr_grid = np.swapaxes(arr_view, 1, 2).reshape(-1, aaa, bbb, c)
print arr_grid.shape

arr_view = np.swapaxes(arr_view, 1, 2)
print arr_view.shape
arr_grid = [arr_view[j] for j in zip(*np.unravel_index(np.arange(a*b//aaa//bbb),
                                                       (a//aaa, b//bbb)))]

for i in arr_grid:
    print i.shape

n_cubes = 10

I_red = I[(I.shape[0] % n_cubes)/2:-(I.shape[0] % n_cubes)/2, (I.shape[1] % n_cubes)/2:-(I.shape[1] % n_cubes)/2, (I.shape[2] % n_cubes)/2:-(I.shape[2] % n_cubes)/2]

print I_red.shape

grids = []
for i in range(0, I_red.shape[1], I_red.shape[1]/n_cubes):
    grids.append(I_red[:, i:i+I_red.shape[1]/n_cubes, :])
for i in grids:
    print i.shape

for i in grids:
    I_arr = np.arange(i.shape[0]*i.shape[1]*i.shape[2]).reshape(i.shape[0], i.shape[1], i.shape[2])
    print I_arr.shape
    I_arr_view = I_arr.reshape(i.shape[0]//(i.shape[0]/n_cubes), i.shape[0]/n_cubes, i.shape[1], i.shape[2]//(i.shape[2]/n_cubes), i.shape[2]/n_cubes)
    print I_arr_view.shape
    I_arr_view = np.swapaxes(arr_view, 1, 3).reshape(-1, i.shape[0]/n_cubes, i.shape[1], i.shape[2]/n_cubes)
    print I_arr_view.shape