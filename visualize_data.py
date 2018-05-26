import numpy as np
import matplotlib.pyplot as plt
import fetch_data

data_training, data_testing = fetch_data.load_CIFAR_dataset(shuffle=True, percentage=100)
training_labels = np.array(data_training)[:, 1]
class_name = fetch_data.load_CIFAR_classnames()

number_classes = 10
number_examples = 10

img_set = [np.where(training_labels == class_id)[0][0:number_examples] for class_id in range(number_classes)]

fig, axarr = plt.subplots(number_classes, number_examples + 1, figsize=(10, 10))
fig.suptitle('Random images from each class in the CIFAR-10 dataset')

for class_id in range(number_classes):
    for i in range(number_examples):
        axarr[class_id, i].imshow(data_training[img_set[class_id][i]][0])
        axarr[class_id, i].axis('off')
    axarr[class_id, number_examples].text(0.5, 0.5, class_name[class_id], fontsize=10, horizontalalignment='center',
                                          verticalalignment='center')
    axarr[class_id, number_examples].axis("off")

plt.show()
