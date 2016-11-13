import numpy as np
import nibabel as nib

a, b, c = 9, 9, 3
aaa, bbb = 3, 3
arr = np.arange(a*b*c).reshape(a, b, c)

print arr.shape

arr_view = arr.reshape(a//aaa, aaa, b//bbb, bbb, c)

print arr_view.shape

arr_view = np.swapaxes(arr_view, 1, 2)

print arr_view.shape
arr_grid = [arr_view[j] for j in zip(*np.unravel_index(np.arange(a*b//aaa//bbb),
                                                       (a//aaa, b//bbb)))]


arr_sum = [np.sum(i) for i in arr_grid]
print arr_sum

Targets = np.genfromtxt("data/targets.csv")

X_train = []
for i in range(1, 2):
    example = nib.load("data/set_train/train_"+str(i)+".nii")
    image = example.get_data()
I = image[:,:,:,0]

n_cubes = 10
print I.shape

I_new = I[(176%n_cubes)/2:-(176%n_cubes)/2, (208%n_cubes)/2:-(208%n_cubes)/2, (176%n_cubes)/2:-(176%n_cubes)/2]

print I_new.shape
I_view = I_new.reshape(176//(176/n_cubes), 176/n_cubes, 208//(208/n_cubes), 208/n_cubes, 176//(176/n_cubes), 176/n_cubes)

print I_view.shape

I_view = np.swapaxes(I_view,1,2)
I_grid = [I_view[j] for j in zip(*np.unravel_index(np.arange(((176/n_cubes*n_cubes)*(208/n_cubes*n_cubes))//n_cubes//n_cubes),
                                                       (170//n_cubes, 200//n_cubes)))]

print I_grid