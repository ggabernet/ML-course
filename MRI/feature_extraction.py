from skimage import measure
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

class Contours:
    def __init__(self, intensity, min_size):
        self.contours = []
        self.intensity = intensity
        self.min_size = min_size

    def get_2D_contours(self, X):
        self.contours = []
        for n in len(X):
            all_contours = measure.find_contours(X[n])
            sample_contours = []
            for c in all_contours:
                if c.shape[0] > self.min_size:
                    sample_contours.append(c)
            self.contours.append(sample_contours)
        return self.contours

    def plot_2D_contours(self, X):
        for n in range(len(X)):
            fig, ax = plt.subplots()
            ax.imshow(X[n], interpolation='nearest', cmap=plt.cm.gray)

            for contour in self.contours[n]:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

            ax.axis('image')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.show()