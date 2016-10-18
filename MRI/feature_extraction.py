from skimage import measure
import matplotlib.pyplot as plt
import numpy as np


class Intensities:
    def __init__(self, layers_x_dim):
        self.layers_x_dim = layers_x_dim

    def calculate_descriptor(self, X):
        descriptor= []
        for n in X:
            xlay = np.linspace(1, n.shape[0] - 1, self.layers_x_dim, dtype=int)
            ylay = np.linspace(1, n.shape[1] - 1, self.layers_x_dim, dtype=int)
            zlay = np.linspace(1, n.shape[2] - 1, self.layers_x_dim, dtype=int)
            desc = []
            for x in xlay:
                layer = n[x, :, :]
                desc.append(self._get_layer_intensity_sum(layer))
            for y in ylay:
                layer = n[:, y, :]
                desc.append(self._get_layer_intensity_sum(layer))
            for z in zlay:
                layer = n[:, :, z]
                desc.append(self._get_layer_intensity_sum(layer))
            descriptor.append(desc)
        self.descriptor = np.asarray(descriptor)
        return self

    def _get_layer_intensity_sum(self, layer):
        layerArray = np.asarray(layer)
        layerFlat = layerArray.flatten(order='C')
        intensity = np.sum(layerFlat)/np.size(layerFlat)
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


