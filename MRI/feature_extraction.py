from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import sobel, scharr, prewitt


class CenterCut:
    def __init__(self):
        self.cut = []
    def make_cut(self, X):
        cut = []
        for n in X:
            cut.append(n[50:120,50:150,50:100])
        self.cut = cut
        return self


class Covariance:
    def __init__(self):
        self.cov_matrix = []

    def calculate_covariance(self, X, y):
        data = []
        for n in X:
            data.append(n.flatten(order='C'))
        data = np.asarray(data)
        cov_mat = []
        for i in range(1,data.shape[1]):
            cov = np.mean(np.dot(data[i,:],y))-np.mean(data[i,:])*np.mean(y)
            cov_mat.append(cov)
        self.cov_matrix = cov_mat
        return self


class Filtering:
    def __init__(self):
        self.transformed = []

    def calculate_prewitt(self, X):
        trans = []
        for n in X:
            p = []
            for i in range(0, n.shape[2]):
                layer=n[:,:,i]
                p.append(prewitt(layer))
            trans.append(np.asarray(p))
        self.transformed = np.asarray(trans)
        return self

    def calculate_scharr(self, X, axis=0):
        axis = axis
        trans = []
        for n in X:
            p = []
            for i in range(0, n.shape[axis]):
                layer = n[:, :, i]
                p.append(scharr(layer))
            trans.append(np.asarray(p))
        self.transformed = np.asarray(trans)
        return self

    def calculate_sobel(self, X, axis=0):
        axis = axis
        trans = []
        for n in X:
            p = []
            for i in range(0, n.shape[axis]):
                layer = n[:, :, i]
                p.append(sobel(layer))
            trans.append(np.asarray(p))
        self.transformed = np.asarray(trans)
        return self

    def flatten(self, X):
        flat = []
        for n in X:
            flat.append(n.flatten(order='C'))
        return np.asarray(flat)

# TODO: add gaussian preprocessing


class Intensities:
    def __init__(self):
        self.descriptor = []

    def calculate_intensity_layers(self, X, layers_x_dim):
        self.layers_x_dim = layers_x_dim
        descriptor= []
        for n in X:
            xlay = np.linspace(1, n.shape[0] - 1, self.layers_x_dim, dtype=int)
            ylay = np.linspace(1, n.shape[1] - 1, self.layers_x_dim, dtype=int)
            zlay = np.linspace(1, n.shape[2] - 1, self.layers_x_dim, dtype=int)
            desc = []
            for x in xlay:
                layer = n[x, :, :]
                desc.append(self._get_array_intensity_sum(layer))
            for y in ylay:
                layer = n[:, y, :]
                desc.append(self._get_array_intensity_sum(layer))
            for z in zlay:
                layer = n[:, :, z]
                desc.append(self._get_array_intensity_sum(layer))
            descriptor.append(desc)
        self.descriptor = np.asarray(descriptor)
        return self

    def calculate_intensity_cubes(self, X, size_cubes, iterations=5):
        it = iterations
        size_cubes = size_cubes
        descriptor = []
        cubes_edge = it * 2 - 1
        for n in X:
            # first center cube
            x1 = n.shape[0] / 2 - size_cubes * cubes_edge / 2
            x2 = n.shape[0] / 2 + size_cubes * cubes_edge / 2
            y1 = n.shape[1] / 2 - size_cubes * cubes_edge / 2
            y2 = n.shape[1] / 2 + size_cubes * cubes_edge / 2
            z1 = n.shape[2] / 2 - size_cubes * cubes_edge / 2
            z2 = n.shape[2] / 2 + size_cubes * cubes_edge / 2
            int_cubes = []
            for i in range(x1, x2, size_cubes):
                for j in range(y1, y2, size_cubes):
                    for k in range(z1, z2, size_cubes):
                        cube = n[i:i + size_cubes, j:j + size_cubes, k:k + size_cubes]
                        int_cubes.append(self._get_array_intensity_sum(cube))
            descriptor.append(int_cubes)
        self.descriptor = np.asarray(descriptor)
        return self

    def _get_array_intensity_sum(self, array):
        arrArray = np.asarray(array)
        arrFlat = arrArray.flatten(order='C')
        intensity = np.sum(arrFlat)/np.size(arrFlat)
        return intensity


class Contours:
    def __init__(self, intensity, min_size, layers_x_dim):
        self.intensity = intensity
        self.min_size = min_size
        self.layers_x_dim = layers_x_dim
        self.descriptor = []

    def calculate_descriptor(self, X):
        descriptor = []

        for n in X:
            xlay = np.linspace(1, n.shape[0] - 1, self.layers_x_dim, dtype=int)
            ylay = np.linspace(1, n.shape[1] - 1, self.layers_x_dim, dtype=int)
            zlay = np.linspace(1, n.shape[2] - 1, self.layers_x_dim, dtype=int)
            desc = []
            for x in xlay:
                layer = n[x, :, :]
                contours = self._get_layer_contours(layer)
                desc.append(self._get_contour_number(contours))
                #desc.append(self._get_biggest_contour_area(contours))
                desc.append(self._get_biggest_minus_n_biggest_contour_areas(contours, 1))
                desc.append(self._get_biggest_minus_n_biggest_contour_areas(contours, 2))
                desc.append(self._get_biggest_minus_n_biggest_contour_areas(contours, 3))
            for y in ylay:
                layer = n[:, y, :]
                contours = self._get_layer_contours(layer)
                desc.append(self._get_contour_number(contours))
                #desc.append(self._get_biggest_contour_area(contours))
                desc.append(self._get_biggest_minus_n_biggest_contour_areas(contours, 1))
                desc.append(self._get_biggest_minus_n_biggest_contour_areas(contours, 2))
                desc.append(self._get_biggest_minus_n_biggest_contour_areas(contours, 3))
            for z in zlay:
                layer = n[:, :, z]
                contours = self._get_layer_contours(layer)
                desc.append(self._get_contour_number(contours))
                #desc.append(self._get_biggest_contour_area(contours))
                desc.append(self._get_biggest_minus_n_biggest_contour_areas(contours, 1))
                desc.append(self._get_biggest_minus_n_biggest_contour_areas(contours, 2))
                desc.append(self._get_biggest_minus_n_biggest_contour_areas(contours, 3))
            descriptor.append(desc)
        self.descriptor = np.asarray(descriptor)
        return self

    def _get_layer_contours(self, layer):
        layer_contours_ = []
        all_contours = measure.find_contours(layer, self.intensity)
        for c in all_contours:
            if c.shape[0] > self.min_size:
                layer_contours_.append(c)
        return layer_contours_

    def _get_contour_number(self, contours):
        return len(contours)

    def _get_biggest_contour_area(self, contours):
        areas = []
        for contour in contours:
            x_coord = np.asarray(contour[:, 0])
            y_coord = np.asarray(contour[:, 1])
            y_next = np.roll(y_coord, -1)
            y_diff = [abs(x) for x in (y_next - y_coord)]
            area = np.sum(np.prod([x_coord, y_diff], axis=0))
            areas.append(area)
        if areas:
            a = np.max(areas)
        else:
            a = 0
        return a

    def _get_biggest_minus_second_biggest_contour_area(self, contours):
        areas = []
        for contour in contours:
            x_coord = np.asarray(contour[:, 0])
            y_coord = np.asarray(contour[:, 1])
            y_next = np.roll(y_coord, -1)
            y_diff = [abs(x) for x in (y_next - y_coord)]
            area = np.sum(np.prod([x_coord, y_diff], axis=0))
            areas.append(area)
        if areas:
            a = np.max(areas)
            smallerareas = areas.remove(np.max(areas))
            b = np.max(smallerareas)
            if b:
                area = a-b
            else:
                area = a
        else:
            area = 0
        return area

    def _get_biggest_minus_n_biggest_contour_areas(self, contours, n):
        areas = []
        for contour in contours:
            x_coord = np.asarray(contour[:, 0])
            y_coord = np.asarray(contour[:, 1])
            y_next = np.roll(y_coord, -1)
            y_diff = [abs(x) for x in (y_next - y_coord)]
            area = np.sum(np.prod([x_coord, y_diff], axis=0))
            areas.append(area)
        i = 0
        while i < n:
            if areas:
                a = np.max(areas)
                areas.remove(np.max(areas))
                if areas:
                    b = np.max(areas)
                    area = a - b
                else:
                    area = a
            else:
                area = 0
            i += 1
        return area


    def _plot_layer_2D_contours(self, X):
        for n in range(len(X)):
            fig, ax = plt.subplots()
            ax.imshow(X[n], interpolation='nearest', cmap=plt.cm.gray)

            for contour in self.contours[n]:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

            ax.axis('image')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.show()


