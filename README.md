# Image_Classification
Data Scientist Exercise

First, in fetch_data you can download data(if it doesn't exist) and load data and class name
(if you don't have GPU, you can set percentage in load_CIFAR_dataset to 10).

In visualize_data I showed an image that has 10 images from any class in the dataset.

In HoG, I used HoG algorithm to serve as a benchmark, for this purpose first I changed RGB photo to gray photo and then I used feature extraction, In the end, I used linearSVC for classification.

I used inception for pre-train CNN, first must be load model and then use Tensorflow to create the graph of the model, then I used this network for feature extraction on data, after running this network I have 2048 feature for an image. In the end, I save features training, testing and their labels.

In visualize CNN, I used PCA for dimension reduction.

In the end, I used linearSVC for image classification. I check c from 0.0005 to 1.0 with a step of 0.0005 and found 0.005 was the best value for C.
