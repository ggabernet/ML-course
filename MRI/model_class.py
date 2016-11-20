import nibabel as nib
from sklearn.svm import SVR
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
            self.dim=[dim_1, dim_2]
        elif self.dimension_option == '3D':
            dim_1=176
            dim_2=208
            dim_3=176
            self.image_dimension = dim_1*dim_2*dim_3
            self.dim = [dim_1, dim_2, dim_3]
        else:
            self.dimension_option='2D' # set a default option
            dim_1=176
            dim_2=208
            self.image_dimension = dim_1*dim_2
            self.dim=[dim_1, dim_2]

    def image_to_data(self, set_option):
        #Data set option specification
        if set_option == 'Train':
            path="data/set_train/train_"
            self.Number_of_images=278
        if set_option == 'Test':
            path="data/set_test/test_"
            self.Number_of_images=138
        #set output dimensions by the number of images
        test_image_size=np.zeros(self.dim)
        print test_image_size.shape
        [w, l] = test_image_size[20:160,1:207].shape
        self.image_dimension=w*l
        print w, l
        image_data = np.zeros((self.image_dimension, self.Number_of_images))
        #Image loading and data processing
        for i in range(1, self.Number_of_images + 1):
            print "Current image is :", str(i)
            nib_image = nib.load(path+str(i)+".nii")
            nib_image_smoothed=processing.smooth_image(nib_image, self.fwhm)
            image = nib_image_smoothed.get_data()
            if self.dimension_option == '2D':
                I = image[:, :, self.slice, 0]
                I = np.asarray(I, dtype=float)
            elif self.dimension_option == '3D':
                I = image[:, :, :, 0]
                I=np.asarray(I, dtype=float)
                #scale data
            I=I[20:160,1:207]
            I=I/np.max(I)
            #Image processing
            I=prewitt(I) #Edge detection
            I = gaussian(I, sigma=self.gauss) #Gaussian blurring of the edges
            Iflat=I.flatten(order='C') # Data flattening
            image_data[:,i-1]=Iflat
        #image_data=np.transpose(image_data)
        print 'image to data : step complete'
        return image_data

    def target_to_data(self):
        Targets = np.genfromtxt("data/targets.csv")
        print 'target to data : step complete'
        return Targets

    def data_processing(self, input_img, input_target, threshold):
        Input_data = np.asarray(input_img)
        Target_data=np.asarray(input_target)
        Covarience=np.zeros((self.image_dimension))
        ignore_index=[]
        for i in range(1,self.image_dimension):
            cov = np.mean(np.dot(Input_data[i, :], Target_data)) - np.mean(Input_data[i, :]) * np.mean(Target_data)
            if cov > threshold:
                Covarience[i]=cov
            else:
                ignore_index.append(i)
        #check image
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
        X_train, X_test, y_train, y_test = train_test_split(validate_input_data, validate_output_data, test_size=prop_test, random_state=42)
        # regression machine classifier
        clf=linear_model.Lasso(alpha=linear_alpha)
        # regression machine classifier
        regr = Pipeline([('scl', StandardScaler()),
                         ('pca', PCA(n_components=n_pca)),
                         ('clf',clf)])

        regr.fit(X_train, y_train)

        # The mean squared error
        print("Mean squared error: %.2f"
              % np.mean((regr.predict(X_test) - y_test) ** 2))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % regr.score(X_test, y_test))


        #Plotiting options
        plt.scatter(y_train, regr.predict(X_train), color='Blue')
        plt.scatter(y_test, regr.predict(X_test), color='Red')

        axes = plt.gca()
        axes.set_xlim([0,int(np.max(y_train)+10)])
        axes.set_ylim([0,int(np.max(y_train)+10)])

        plt.xlabel('Actual age values')
        plt.ylabel('Predicted age values')

        plt.plot()
        plt.show()
        print 'cross validation  : step complete'
        return regr

    def cross_validation(self, validate_input_data, validate_output_data, linear_alpha, n_pca, n_kfold):
        kf=KFold(n_splits=n_kfold)

        clf=linear_model.Lasso(alpha=linear_alpha)
        # regression machine classifier
        regr = Pipeline([('scl', StandardScaler()),
                         ('pca', PCA(n_components=n_pca)),
                         ('clf',clf)])
        # Train the model using the training setskf=KF
        for train, test in kf.split(validate_input_data,validate_output_data):
            X_train=validate_input_data[train]
            y_train=validate_output_data[train]
            X_test=validate_input_data[test]
            y_test=validate_output_data[test]

            regr.fit(X_train, y_train)

            # The mean squared error
            print("Mean squared error: %.2f"
                  % np.mean((regr.predict(X_test) - y_test) ** 2))
            # Explained variance score: 1 is perfect prediction
            print('Variance score: %.2f' % regr.score(X_test, y_test))


            #Plotiting options
            plt.scatter(y_train, regr.predict(X_train), color='Blue')
            plt.scatter(y_test, regr.predict(X_test), color='Red')

            axes = plt.gca()
            axes.set_xlim([0,int(np.max(y_train)+10)])
            axes.set_ylim([0,int(np.max(y_train)+10)])

            plt.xlabel('Actual age values')
            plt.ylabel('Predicted age values')

            plt.plot()
            plt.show()
        print 'cross validation  : step complete'
        return regr

    def apply_model(self, test_option):
        Train_image_data=self.image_to_data('Train')
        Train_target_data = self.target_to_data()
        ignore_index=self.data_processing(Train_image_data,Train_target_data, 50)
        Train_filtered=self.filter_pixels(Train_image_data,ignore_index)
        regr = self.validation(Train_filtered, Train_target_data, 0.001, 10, 0.33)
        #regr=self.cross_validation(Train_filtered,Train_target_data, 0.01, 10, 10)

        if test_option == 0:
            Test_image_data = self.image_to_data('Test')
            Test_image_filtered= self.filter_pixels(Test_image_data,ignore_index)

            output=open('Submission.csv','w+')
            output.write("ID,Prediction"+'\n')

            for idx, line in enumerate(regr.predict(Test_image_filtered)):
                print line
                output.write(str(idx+1)+','+str(line)+'\n')
            output.close()
        print 'apply model : step complete'

#Runs the model with the input parameters
x=ML_brain('2D', 80, 1, 1, 50)
x.apply_model(0)





