from skimage import measure
import matplotlib.pyplot as plt
import numpy as np

class Contours:
    def __init__(self, intensity, min_size, layers_x_dim):
        self.intensity = intensity
        self.min_size = min_size
        self.layers_x_dim = layers_x_dim
        self.contours = []
        self.n_contours = []
        self.biggest_cluster_size = []
        self.descriptor = []


    def get_layer_contours_(self, layer):
        layer_contours_ = []
        all_contours = measure.find_contours(layer, self.intensity)
        for c in all_contours:
            if c.shape[0] > self.min_size:
                layer_contours_.append(c)
        return layer_contours_

    def get_layer_contour_number_(self, layer):
        contours = self.get_layer_contours_(layer)
        n_contours = len(contours)
        return n_contours

    def get_contour_area_(self, contour):
        x_coord = np.asarray(c[:, 0])
        y_coord = np.asarray(c[:, 1])
        y_next = np.roll(y_coord, -1)
        y_diff = [abs(x) for x in (y_next - y_coord)]

        return np.sum(np.prod([x_coord, y_diff], axis=0))

    def plot_layer_2D_contours_(self, X):
        for n in range(len(X)):
            fig, ax = plt.subplots()
            ax.imshow(X[n], interpolation='nearest', cmap=plt.cm.gray)

            for contour in self.contours[n]:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

            ax.axis('image')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.show()


