import numpy as np
from sklearn import decomposition
from matplotlib import pyplot as plt

data = np.load('data/CIFAR10_features.npz')

training_data_x = data['features_training']
training_data_y = data['labels_training']

testing_data_x = data['features_testing']
testing_data_y = data['labels_testing']

pca = decomposition.PCA(n_components=2)
pca.fit(training_data_x)

X = pca.transform(training_data_x)

plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=training_data_y, cmap='tab10')
plt.show()
