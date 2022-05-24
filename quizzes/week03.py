"""
Vedrana Andersen Dahl (vand@dtu.dk) 
Anders Bjorholm Dahl (abda@dtu.dk)
"""

import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import local_features as lf
import scipy.ndimage


def ind2labels(ind):
    """ Helper function for transforming uint8 image into labeled image."""
    return np.unique(ind, return_inverse=True)[1].reshape(ind.shape)

path = './data/' # Change path to your directory

#%% READ IN IMAGES
training_image = skimage.io.imread(path + 'training_image.png')
training_image = training_image.astype(np.float)
training_labels = skimage.io.imread(path + 'training_labels.png')

training_labels = ind2labels(training_labels)
nr_labels = np.max(training_labels)+1 # number of labels in the training image

fig, ax = plt.subplots(1,2)
ax[0].imshow(training_image, cmap=plt.cm.gray)
ax[0].set_title('training image')
ax[1].imshow(training_labels)
ax[1].set_title('labels for training image')

#%% TRAING THE MODEL

sigma = [1,2,3]
features = lf.get_gauss_feat_multi(training_image, sigma)
features = features.reshape((features.shape[0], features.shape[1]*features.shape[2]))
labels = training_labels.ravel()

labels = training_labels.ravel()


nr_keep = 15000 # number of features randomly picked for clustering 
keep_indices = np.random.permutation(np.arange(features.shape[0]))[:nr_keep]

features_subset = features[keep_indices,:]
labels_subset = labels[keep_indices]

nr_clusters = 1000 # number of feature clusters
# for speed, I use mini-batches
kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=nr_clusters, batch_size=2*nr_clusters)
kmeans.fit(features_subset)
assignment = kmeans.labels_

edges = np.arange(nr_clusters+1)-0.5 # histogram edges halfway between integers
hist = np.zeros((nr_clusters,nr_labels))
for l in range(nr_labels):
    hist[:,l] = np.histogram(assignment[labels_subset==l],bins=edges)[0]
sum_hist = np.sum(hist,axis=1)
cluster_probabilities = hist/(sum_hist.reshape(-1,1))

fig, ax = plt.subplots(1,2)
legend_label = [f'label {x}' for x in range(nr_labels)]

ax[0].plot(hist,'.')
ax[0].set_xlabel('cluster id')
ax[0].set_ylabel('number of features in cluster')
ax[0].legend(legend_label)
ax[0].set_title('features in clusters per label')
ax[1].plot(cluster_probabilities,'.')
ax[1].set_xlabel('cluster id')
ax[1].set_ylabel('label probability for cluster')
ax[1].legend(legend_label)
ax[1].set_title('cluster probabilities')

# Finished training

#%% USING THE MODEL
testing_image = skimage.io.imread(path + 'testing_image.png')
testing_image = testing_image.astype(np.float)

features_testing = lf.get_gauss_feat_multi(testing_image, sigma)
features_testing = features_testing.reshape((features_testing.shape[0], features_testing.shape[1]*features_testing.shape[2]))
labels = training_labels.ravel()

assignment_testing = kmeans.predict(features_testing)

probability_image = np.zeros((assignment_testing.size, nr_labels))
for l in range(nr_labels):
    probability_image[:,l] = cluster_probabilities[assignment_testing, l]
probability_image = probability_image.reshape(testing_image.shape + (nr_labels,))

P_rgb = np.zeros(probability_image.shape[0:2]+(3,))
k = min(nr_labels,3)
P_rgb[:,:,:k] = probability_image[:,:,:k]

fig, ax = plt.subplots(1,2)
ax[0].imshow(testing_image, cmap=plt.cm.gray)
ax[0].set_title('testing image')
ax[1].imshow(P_rgb)
ax[1].set_title('probabilities for testing image as RGB')

#%% SMOOTH PROBABILITY MAP

sigma = 3 # Gaussian smoothing parameter

seg_im_max = np.argmax(P_rgb,axis = 2)
c = np.eye(P_rgb.shape[2])
P_rgb_max = c[seg_im_max]

probability_smooth = np.zeros(probability_image.shape)
for i in range(0,probability_image.shape[2]):
    probability_smooth[:,:,i] = scipy.ndimage.gaussian_filter(probability_image[:,:,i],sigma,order=0)
seg_im_smooth = np.argmax(probability_smooth,axis=2)

probability_smooth_max = c[seg_im_smooth]

P_rgb_smooth = np.zeros(probability_smooth_max.shape[0:2]+(3,))
k = min(nr_labels,3)
P_rgb_smooth[:,:,:k] = probability_smooth[:,:,:k]
P_rgb_smooth_max = np.zeros(probability_smooth_max.shape[0:2]+(3,))
P_rgb_smooth_max[:,:,:k] = probability_smooth_max[:,:,:k]

# Display result
fig,ax = plt.subplots(2,4,sharex=True,sharey=True)
ax[0][0].imshow(P_rgb[:,:,0])
ax[0][1].imshow(P_rgb[:,:,1])
ax[0][2].imshow(P_rgb[:,:,2])
ax[0][3].imshow(P_rgb_max)
ax[1][0].imshow(P_rgb_smooth[:,:,0])
ax[1][1].imshow(P_rgb_smooth[:,:,1])
ax[1][2].imshow(P_rgb_smooth[:,:,2])
ax[1][3].imshow(P_rgb_smooth_max)
