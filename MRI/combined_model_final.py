import nibabel as nib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

#------------------------
# PRE-PROCESSING CLASS
#------------------------

class CenterCutCubes(BaseEstimator, TransformerMixin):
    """
    Class (inherited from sklearn base class BaseEstimator) to select a cube of voxels, and apply a coarse-grain transformation.
    """
    def __init__(self, size_cubes, x1=50, x2=120, y1=80, y2=150, z1=50, z2=100):
        self.cut = []
        self.descriptor = []
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.z1 = z1
        self.z2 = z2
        self.size_cubes = size_cubes

    def fit(self, X_train, y=None):
        return self

    def transform(self, X_train, y=None):
        # selecting the x, y, z dimensions of the cube to extract (maximum variance zone)
        cut = []
        for n in X_train:
            cut.append(n[self.x1:self.x2, self.y1:self.y2, self.z1:self.z2])
        self.cut = cut
        descriptor = []
        # Performing coarse-graining by taking the maximum intensity value in cubes of defined dimensions.
        for n in cut:
            int_cubes = []
            for i in range(0, n.shape[0], self.size_cubes):
                for j in range(0, n.shape[1], self.size_cubes):
                    for k in range(0, n.shape[2], self.size_cubes):
                        cube = n[i:i + self.size_cubes, j:j + self.size_cubes, k:k + self.size_cubes]
                        int_cubes.append(self._get_array_intensity_max(cube))
            descriptor.append(int_cubes)

        self.descriptor = np.asarray(descriptor)
        return self.descriptor

    def _get_array_intensity_max(self, array):
        """
        Method to select the voxel with maximum intensity within a cube of voxels.
        """
        arrArray = np.asarray(array)
        arrFlat = arrArray.flatten(order='C')
        intensity = np.max(arrFlat)
        return intensity

# -----------------------
# LOADING TRAINING DATA
# -----------------------

Targets = np.genfromtxt("data/targets.csv")

Data = []
for i in range(1, 279):
    imagefile = nib.load("data/set_train/train_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    imagefile.uncache()
    Data.append(np.asarray(I))


X_train = Data
y_train = Targets


# ---------------------------------------------------
# PIPELINE FOR PRE-PROCESSING AND TRAINING SVR MODEL
#----------------------------------------------------

pipe = Pipeline([('cut', CenterCutCubes(size_cubes=3)), # Extracting cube of maximum variance from 3D brain images and coarse-graining it by taking the maximum intensity of a voxel in a 3x3x3 voxel cube.
                ('scl', StandardScaler()), # Standard scaling of the data
                ('var', VarianceThreshold()), # Removing features that have a variance of zero
                ('pca', PCA(n_components=250)), # Performing PCA and taking top 250 principal components.
                ('clf', SVR(kernel='linear', C=0.1))]) #Training support vector machine with linear kernel and C parameter 0.1.


pipe.fit(X_train, y_train) #Fitting pipeline (includes SVR model)


# -------------------------
# LOADING TEST DATA SET
# -----------------------

Data_test = []
for i in range(1, 139):
    imagefile = nib.load("data/set_test/test_"+str(i)+".nii")
    image = imagefile.get_data()
    I = image[:, :, :, 0]
    Data_test.append(np.asarray(I))

X_test = Data_test


# -------------------------------
# PERFORM PREDICTION ON TEST SET
# -------------------------------

predictions = pipe.predict(X_test)

with open("SubmissionFinal.csv", mode='w') as f:
    f.write("ID,Prediction\n")
    for idx, pred in enumerate(predictions):
        pred = round(pred, 0)
        f.write(str(idx+1)+','+str(pred)+'\n')
