import nilearn as nl
from nilearn.decomposition import DictLearning
from nilearn import plotting
from nilearn.regions import RegionExtractor
import nibabel as nib
import numpy as np
from sklearn.cross_validation import train_test_split
# Import dictionary learning algorithm from decomposition module and call the
# object and fit the model to the functional datasets


Targets = np.genfromtxt("data/targets.csv")

X_train = []
for i in range(1, 30):
    example = nl.image.load_img("data/set_train/train_"+str(i)+".nii")
    X_train.append(example)
Data = np.asarray(X_train)


# Initialize DictLearning object
dict_learn = DictLearning(n_components=5, smoothing_fwhm=6.,
                          memory="nilearn_cache", memory_level=2,
                          random_state=0)
# Fit to the data
print "Fitting dictionary..."
dict_learn.fit(X_train)
# Resting state networks/maps
print "Resting state networks/maps..."
components_img = dict_learn.masker_.inverse_transform(dict_learn.components_)

# Visualization of resting state networks
# Show networks using plotting utilities

plotting.plot_prob_atlas(components_img, view_type='filled_contours',
                         title='Dictionary Learning maps')

# Import Region Extractor algorithm from regions module
# threshold=0.5 indicates that we keep nominal of amount nonzero voxels across all
# maps, less the threshold means that more intense non-voxels will be survived.
print "Extracting regions..."
extractor = RegionExtractor(components_img, threshold=0.5,
                            thresholding_strategy='ratio_n_voxels',
                            extractor='local_regions',
                            standardize=True, min_region_size=1350)
# Just call fit() to process for regions extraction
extractor.fit()
# Extracted regions are stored in regions_img_
regions_extracted_img = extractor.regions_img_
# Each region index is stored in index_
regions_index = extractor.index_
# Total number of regions extracted
n_regions_extracted = regions_extracted_img.shape[-1]

# Visualization of region extraction results
title = ('%d regions are extracted from %d components.'
         '\nEach separate color of region indicates extracted region'
         % (n_regions_extracted, 5))
plotting.plot_prob_atlas(regions_extracted_img, view_type='filled_contours',
                         title=title)
