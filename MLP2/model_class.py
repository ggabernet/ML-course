import nibabel as nib
from sklearn.svm import SVR, SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.decomposition import PCA
import skimage
from skimage.filters import roberts, sobel, scharr, prewitt, gaussian
from sklearn.model_selection import GridSearchCV

from skimage import feature
from skimage import exposure
from nibabel import processing

from scipy import ndimage
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

#class methods (try)
from sklearn.metrics import log_loss
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn.ensemble import RandomForestClassifier

class ML_brain:
    def __init__(self,dimension_selection, slice_number, smoothing_sigma, gauss_sigma, threshold):
        # Image parameters
        self.Number_of_images = 0
        self.dimension_option=dimension_selection
        self.slice=slice_number
        #Smoothing parameters
        self.fwhm = processing.sigma2fwhm(smoothing_sigma)
        self.gauss=gauss_sigma
        #Data parameters
        self.threshold=threshold
        #Data dimension options based on representation selection
        self.reshape_parm=(0,0)
        if self.dimension_option == '2D':
            dim_1=176
            dim_2=208
            self.image_dimension = dim_1*dim_2
            self.dim = [dim_1, dim_2]
            self.test_image = np.zeros((dim_1, dim_2))
        elif self.dimension_option == '3D':
            dim_1=176
            dim_2=208
            dim_3=176
            self.image_dimension = dim_1*dim_2*dim_3
            self.dim = [dim_1, dim_2, dim_3]
            self.test_image = np.zeros((dim_1,dim_2,dim_3))

    def image_to_data(self, set_option):
        #Data set option specification
        if set_option == 'Train':
            path="data/set_train/train_"
            self.Number_of_images=278
        if set_option == 'Test':
            path="data/set_test/test_"
            self.Number_of_images=138

        if self.dimension_option == '2D':
            [row_low, row_up] = [65, 150]
            [column_low, column_up] = [60, 150]
            test_image_change = self.test_image[row_low:row_up, column_low:column_up]
            [w, h] = test_image_change.shape
            self.image_dimension = w * h
            self.dim = [w, h]
        elif self.dimension_option == '3D':
            [row_low, row_up] = [20, 150]
            [column_low, column_up] = [20, 150]
            [width_low, width_up] = [20, 150]
            test_image_change = self.test_image[row_low:row_up, column_low:column_up, width_low:width_up]
            [w, h, l] = test_image_change.shape
            self.image_dimension = w * h * l
            self.dim = [w, h, l]

        image_data = np.zeros((self.image_dimension, self.Number_of_images))

        #Image loading and data processing
        for i in range(1, self.Number_of_images + 1):
            print "Current image is :", str(i)
            nib_image = nib.load(path+str(i)+".nii")
            nib_image=processing.smooth_image(nib_image, self.fwhm)
            image = nib_image.get_data()
            if self.dimension_option == '2D':
                I = image[:, :, self.slice, 0]
                I = np.asarray(I, dtype=float)
                I = I[row_low:row_up, column_low:column_up]
                # scale data
                I = I / np.max(I)
                # Image processing
                #I = prewitt(I)  # Edge detection
                #I = gaussian(I, sigma=self.gauss)  # Gaussian blurring of the edges
                I = I / np.max(I)
            elif self.dimension_option == '3D':
                I = image[:, :, :, 0]
                I = I[row_low:row_up, column_low:column_up, width_low:width_up]
                #I = ndimage.prewitt(I, axis=-1)
                I=np.asarray(I, dtype=float)
                I = I / np.max(I)
            Iflat=I.flatten(order='C') # Data flattening
            image_data[:,i-1]=Iflat
        #image_data=np.transpose(image_data)
        print 'image to data : step complete'
        return image_data

    def target_to_data(self):
        Targets = np.genfromtxt("data/targets.csv")
        print 'target to data : step complete'
        return Targets

    def data_processing(self, input_img, input_target):
        Input_data = np.asarray(input_img)
        Target_data=np.asarray(input_target)
        Covarience=np.zeros((self.image_dimension))
        ignore_index=[]
        for i in range(1,self.image_dimension):
            cov = np.mean(np.dot(Input_data[i, :], Target_data)) - np.mean(Input_data[i, :]) * np.mean(Target_data)
            if cov > self.threshold:
                Covarience[i]=cov
            else:
                ignore_index.append(i)
        #check image - only for 2D matrix
        #plt.imshow(Covarience.reshape(self.dim))
        #plt.show()
        print 'data processing : step complete'
        return ignore_index

    def filter_pixels(self, image_data, ignore_index):
        filtered_data=np.delete(image_data, ignore_index ,axis=0)
        filtered_data=np.transpose(filtered_data)
        print 'filter pixels : step complete'
        return filtered_data

    def validation(self, validate_input_data, validate_output_data, linear_alpha, n_pca, prop_test):

        X_train, X_test, y_train, y_test = train_test_split(validate_input_data, validate_output_data, test_size=prop_test, random_state=42, stratify=validate_output_data)

        #X_train = validate_input_data
        #y_train = validate_output_data
        # regression machine classifier

        #clf=linear_model.Lasso(alpha=linear_alpha)
        #clf = linear_model.LogisticRegression()
        #clf = LinearSVC()
        #clf = QuadraticDiscriminantAnalysis()


        #kernel = 1.0 * RBF([2.0])
        #clf = GaussianProcessClassifier()

        clf = linear_model.LogisticRegression()
        #clf = GaussianNB()

        #regression machine classifier
        regr = Pipeline([('scl', StandardScaler()),
                          ('pca', PCA(n_components=n_pca)),
                         ('clf', clf)])

        regr.fit(X_train, y_train)

        print "Log Loss train:", log_loss(y_train, regr.predict_proba(X_train))
        print "Log Loss test:", log_loss(y_test,regr.predict_proba(X_test))

        print 'cross validation  : step complete'
        return regr

    def apply_model(self, test_option):
        Train_image_data=self.image_to_data('Train')
        Train_target_data = self.target_to_data()
        ignore_index=self.data_processing(Train_image_data,Train_target_data)
        Train_filtered=self.filter_pixels(Train_image_data,ignore_index)
        regr = self.validation(Train_filtered, Train_target_data, 0.2, 7, 0.33)
        #validation(self, validate_input_data, validate_output_data, linear_alpha, n_pca, prop_test):

        #regr=self.cross_validation(np.transpose(Train_image_data),Train_target_data, 5, 250, 10)

        if test_option == 0:
            Test_image_data = self.image_to_data('Test')
            Test_image_filtered= self.filter_pixels(Test_image_data,ignore_index)

            output=open('Submission.csv','w+')
            output.write("ID,Prediction"+'\n')

            for idx, line in enumerate(regr.predict_proba(Test_image_filtered)):
                print line[1]
                output.write(str(idx+1)+','+str(line[1])+'\n')
            output.close()
        print 'apply model : step complete'

#Runs the model with the input parameters
x=ML_brain('2D', 80, 1, 0, 50)
#x=ML_brain(dimension_selection, slice_number, smoothing_sigma, gauss_sigma, threshold)
x.apply_model(0)
#x.apply_model(test_option)




