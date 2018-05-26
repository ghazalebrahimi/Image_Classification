import fetch_data
from skimage.feature import hog
import numpy as np
from sklearn.svm import LinearSVC
import cv2


training_data, testing_data = fetch_data.load_CIFAR_dataset(shuffle=True, percentage=100)

class_names = fetch_data.load_CIFAR_classnames()

n_training = len(training_data)
n_testing = len(testing_data)


def rgb_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


data_training_gray = [rgb_to_gray(training_data[i][0]) for i in range(n_training)]
data_testing_gray = [rgb_to_gray(testing_data[i][0]) for i in range(n_testing)]

normalize = True
block_norm = 'L1'
orientations = 9
pixels_per_cell = [8, 8]
cells_per_block = [2, 2]


def extractFeature(img, vis=False):
    return hog(img, orientations, pixels_per_cell, cells_per_block, block_norm, visualise=vis, transform_sqrt=normalize)


data_training_x = np.array([extractFeature(data_training_gray[i], vis=False) for i in range(n_training)])
data_training_y = np.array([training_data[i][1] for i in range(n_training)])

data_testing_x = np.array([extractFeature(data_testing_gray[i], vis=False) for i in range(n_testing)])
data_testing_y = np.array([testing_data[i][1] for i in range(n_testing)])


nfeatures = extractFeature(data_training_gray[0], vis=False).size
print('Number of features = {}'.format(nfeatures))


C = 0.9
clf = LinearSVC(C=C)
clf.fit(data_training_x, data_training_y)
print("The score is:", clf.score(data_testing_x, data_testing_y))
