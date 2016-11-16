MLP1
————
Team name: GAL-9000
Team members: Alexander Luke Button, Leon Bichmann, Gisela Gabernet Garriga

Approach followed:

1) Pre-processing:
    - extracted a central prism of voxels that we observed to have maximum variation with age, to avoid the noise of the edges and select the important region of the brain for our problem.
    - coarse-grained the prism by taking sliding cubes of size 3x3x3 and selecting the pixel with maximum intensity within the cube, to reduce dimensionality and noise.
    - flattened the prism to a 1D vector, to be able to train the model.
    - scaled the data with a standard scaler (sklearn StandardScaler), to allow comparability of the images with each other.
    - removed features that showed zero variance across the training data sets (sklearn VarianceThreshold), to remove redundant features.
    - performed PCA analysis, to reduce dimensionality and keep the 250 most relevant features that explain most of the variability of the data.

2) Training Machine-learning model
    - define a SVM (support vector machine) method for prediction of the training data using regression (sklearn SVR)
    - for the SVM we set a linear kernel (best performing with cross-validation) with a regularization parameter of 0.1, regularization prevents over-fitting

3) Post-processing:
    - rounded prediction values to whole numbers as the targets where given like this.