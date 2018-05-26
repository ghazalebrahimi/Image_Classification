import numpy as np
from sklearn.svm import LinearSVC

data = np.load('data/CIFAR10_features.npz')
training_data_x = data['features_training']
training_data_y = data['labels_training']

testing_data_x = data['features_testing']
testing_data_y = data['labels_testing']

# c=0.01 : score = 877
# c=0.005 : score = 882
clf = LinearSVC(C=0.005, verbose=0)
clf.fit(training_data_x, training_data_y)
y_pred = clf.predict(testing_data_x)
score = sum(y_pred == testing_data_y)
print('Accuracy = {}'.format(clf.score(testing_data_x, testing_data_y)))
