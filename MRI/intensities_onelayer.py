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

iterations = 3
size_cubes = 10

cubes = []
#first center cube
x1 = I.shape[0]/2 - size_cubes/2
x2 = I.shape[0]/2 + size_cubes/2
y1 = I.shape[1]/2 - size_cubes/2
y2 = I.shape[1]/2 + size_cubes/2
z1 = I.shape[2]/2 - size_cubes/2
z2 = I.shape[2]/2 + size_cubes/2

x = (x1,x2)
y = (y1,y2)
z = (z1,z2)

I_center = I[x[0]:x[1], y[0]:y[1], z[0]:z[1]]

cubes.append(I_center)

print I_center.shape
print I_center

# 3x3 cubes
#p
xp1 = x2
xp2 = x2 + size_cubes
xp = (xp1, xp2)
yp1 = y2
yp2 = y2 + size_cubes
yp = (yp1, yp2)
zp1 = z2
zp2 = z2 + size_cubes
zp = (zp1, zp2)
#n
xn1 = x1 - size_cubes
xn2 = x1
xn = (xn1, xn2)
yn1 = y1 - size_cubes
yn2 = y1
yn = (yn1, yn2)
zn1 = z1 - size_cubes
zn2 = z1
zn = (zn1, zn2)

loop_x = [x, xp, xn]
loop_y = [y, yp, yn]
loop_z = [z, zp, zn]

for i in loop_x:
    for j in loop_y:
        for k in loop_z:
            cube = I[i[0]:i[1], j[0]:j[1], k[0]:k[1]]
            cubes.append(cube)

# 5x5 cubes
x1 = xn1
x2 = xp2
y1 = yn1
y2 = yp2
z1 = zn1
z2 = zp2



print len(cubes)
for cube in cubes:
    print cube.shape


