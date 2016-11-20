from skimage import measure
import numpy as np
from skimage.filters import sobel, scharr, prewitt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import scipy
from scipy.stats import pearsonr


class CenterCut:
    def __init__(self):
        self.cut = []
        self.descriptor = []
    def make_cut(self, X, x1=50, x2=120, y1=50, y2=150, z1=50, z2=100):
        cut = []
        for n in X:
            cut.append(n[x1:x2,y1:y2,z1:z2])
        self.cut = cut
        return self
    def make_cubes(self, X, size_cubes):
        descriptor = []
        for n in X:
            int_cubes = []
            for i in range(0, n.shape[0], size_cubes):
                for j in range(0, n.shape[1], size_cubes):
                    for k in range(0, n.shape[2], size_cubes):
                        cube = n[i:i + size_cubes, j:j + size_cubes, k:k + size_cubes]
                        int_cubes.append(self._get_array_intensity_max(cube))
            descriptor.append(int_cubes)

        self.descriptor = np.asarray(descriptor)
        return self

    def _get_array_intensity_sum(self, array):
        arrArray = np.asarray(array)
        arrFlat = arrArray.flatten(order='C')
        intensity = np.sum(arrFlat)/np.size(arrFlat)
        return intensity

    def _get_array_intensity_max(self, array):
        arrArray = np.asarray(array)
        arrFlat = arrArray.flatten(order='C')
        intensity = np.max(arrFlat)
        return intensity


class CenterCutCubes(BaseEstimator, TransformerMixin):
    def __init__(self, size_cubes, x1=50, x2=120, y1=50, y2=150, z1=50, z2=100):
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
        cut = []
        for n in X_train:
            cut.append(n[self.x1:self.x2, self.y1:self.y2, self.z1:self.z2])
        self.cut = cut
        descriptor = []
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

    def _get_array_intensity_sum(self, array):
        arrArray = np.asarray(array)
        arrFlat = arrArray.flatten(order='C')
        intensity = np.sum(arrFlat) / np.size(arrFlat)
        return intensity

    def _get_array_intensity_max(self, array):
        arrArray = np.asarray(array)
        arrFlat = arrArray.flatten(order='C')
        intensity = np.max(arrFlat)
        return intensity

class CheckrPixl:
    def __init__(self):
        self.checker = []
    def make_checker(self, X):
        checker=[]
        for n in X:
            checker2 = []
            for m in range(0,100,3):
                X2 = np.array(n[:,m,:]).flatten(order="C").tolist()
                checker2.append(X2)
            checker2 = np.array(checker2).flatten(order="C").tolist()
            checker.append(checker2)
        self.checker = np.array(checker)
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
            for i in range(0, 50, 3):
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

class CovSel(BaseEstimator, TransformerMixin):
    def __init__(self, cut_off=0.5):
        self.cut_off = cut_off

    def fit(self, X_train, y=None):
        pca = PCA(n_components=X_train.shape[1])
        pca.fit(X_train)
        self.covariance = pca.get_covariance()
        self.importance = pca.components_[1]

        x, y = np.where(self.covariance > self.cut_off)

        keep = []
        trash = []

        for n in x:
            if n not in keep and n not in trash:
                corr_feats = y[np.where(x == n)]
                new_corr_feats = []
                for feat in corr_feats:
                    if feat not in keep and feat not in trash:
                        new_corr_feats.append(feat)
                contrib_corr_feats = self.importance[new_corr_feats]
                max_contrib = new_corr_feats[np.argmax(abs(contrib_corr_feats))]
                if max_contrib not in keep:
                    keep.append(max_contrib)
                for i in new_corr_feats:
                    if i not in keep:
                        trash.append(i)
        self.cov_selected_ = keep

        for m in range(X_train.shape[1]):
            if m not in keep and m not in trash:
                keep.append(m)
        keep.sort()
        self.indices_ = keep

        return self

    def transform(self, X, y=None):
        return X[:, self.indices_]

class Intensities:
    def __init__(self):
        self.descriptor = []

    def calculate_intensity_layers(self, X, layers_x_dim):
        self.layers_x_dim = layers_x_dim
        descriptor = []
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
                        int_cubes.append(self._get_array_intensity_max(cube))
            descriptor.append(int_cubes)
        self.descriptor = np.asarray(descriptor)
        return self

    def calculate_intensity_prism(self, X, size_cubes, ncubes_x, ncubes_y):
        size_cubes = size_cubes
        ncubes_x = ncubes_x
        ncubes_y = ncubes_y
        ncubes_z = ncubes_x
        descriptor = []
        for n in X:
            x1 = n.shape[0] / 2 - size_cubes * ncubes_x / 2
            x2 = n.shape[0] / 2 + size_cubes * ncubes_x / 2
            y1 = n.shape[1] / 2 - size_cubes * ncubes_y / 2
            y2 = n.shape[1] / 2 + size_cubes * ncubes_y / 2
            z1 = n.shape[2] / 2 - size_cubes * ncubes_z / 2
            z2 = n.shape[2] / 2 + size_cubes * ncubes_z / 2
            int_cubes = []
            for i in range(x1, x2, size_cubes):
                for j in range(y1, y2, size_cubes):
                    for k in range(z1, z2, size_cubes):
                        cube = n[i:i + size_cubes, j:j + size_cubes, k:k + size_cubes]
                        int_cubes.append(self._get_array_intensity_max(cube))
            descriptor.append(int_cubes)
        self.descriptor = np.asarray(descriptor)
        return self

    def _get_array_intensity_sum(self, array):
        arrArray = np.asarray(array)
        arrFlat = arrArray.flatten(order='C')
        intensity = np.sum(arrFlat)/np.size(arrFlat)
        return intensity

    def _get_array_intensity_max(self, array):
        arrArray = np.asarray(array)
        arrFlat = arrArray.flatten(order='C')
        intensity = np.max(arrFlat)
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


class pvalselect:
    def __init__(self, cubicled_Xtrain, targets):
        self.cubicled_Xtrain = np.array(cubicled_Xtrain)
        self.index = np.argsort(targets)
        self.zero_count = targets.tolist().count(0)
        self.Xtrain_zero = self.cubicled_Xtrain[self.index][0:self.zero_count]
        self.Xtrain_one = self.cubicled_Xtrain[self.index][self.zero_count:]
        self.nr_feat = self.cubicled_Xtrain.shape[1]

    def compute_pvals(self):
        pvals=[]
        for column in range(0,self.nr_feat):
            pvals.append(scipy.stats.ttest_ind(self.Xtrain_zero[:,column],self.Xtrain_one[:,column])[1])
        return pvals