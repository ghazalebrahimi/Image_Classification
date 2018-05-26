import os
import tarfile
import urllib.request
import pickle
import numpy as np
import sklearn

DATA_DIR_PATH = './data/'
CIFAR_DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR_FOLDERNAME = 'cifar-10-batches-py'
CIFAR_BATCH_SIZE = 10000
os.makedirs(DATA_DIR_PATH, exist_ok=True)

CIFAR_TRAINING_FILENAMES = [
    os.path.join(DATA_DIR_PATH, CIFAR_FOLDERNAME, 'data_batch_%d' % i) for i in range(1, 6)
    ]
CIFAR_TESTING_FILENAMES = [os.path.join(DATA_DIR_PATH, CIFAR_FOLDERNAME, 'test_batch')]


def _get_cifar_file_path(filename=""):

    print(os.path.join(DATA_DIR_PATH, CIFAR_FOLDERNAME, filename))
    return os.path.join(DATA_DIR_PATH, CIFAR_FOLDERNAME, filename)


def _unpickle(filename):

    file_path = _get_cifar_file_path(filename)
    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

    return data


def read_CIFAR_files(filenames):

    dataset = []
    for file in filenames:
        with open(file, 'rb') as fo:
            _dict = pickle.load(fo, encoding='bytes')

        data = _dict[b'data']
        labels = _dict[b'labels']

        for k in range(CIFAR_BATCH_SIZE):
            image = data[k].reshape(3, 32, 32)
            image = np.transpose(image, [1, 2, 0])
            dataset.append([image, labels[k]])

    return dataset


def load_CIFAR_classnames():

    raw = _unpickle(filename="batches.meta")[b'label_names']
    names = [x.decode('utf-8') for x in raw]

    return names


def load_CIFAR_dataset(shuffle=True, percentage=100):

    print("Loading dataset")

    if not os.path.isdir(_get_cifar_file_path()):
        os.makedirs(_get_cifar_file_path(), exist_ok=True)
        filename = CIFAR_DATA_URL.split('/')[-1]
        filepath = os.path.join(DATA_DIR_PATH, filename)

        try:
            print("Downloading file {f}".format(f=CIFAR_DATA_URL))
            fpath, _ = urllib.request.urlretrieve(CIFAR_DATA_URL, filepath)
        except:
            print("Failed to download {f}".format(f=CIFAR_DATA_URL))
            raise

        print('Succesfully download')
        tarfile.open(filepath, 'r:gz').extractall(DATA_DIR_PATH)

    trainingData = read_CIFAR_files(CIFAR_TRAINING_FILENAMES)
    testingData = read_CIFAR_files(CIFAR_TESTING_FILENAMES)

    if shuffle:
        print("Shuffling data")
        trainingData = sklearn.utils.shuffle(trainingData)
        testingData = sklearn.utils.shuffle(testingData)

    print("We use {d} percentage of data".format(d=percentage))
    trainingData = trainingData[0:int(len(trainingData) * percentage/100)]
    testingData = testingData[0:int(len(testingData) * percentage/100)]

    return trainingData, testingData
