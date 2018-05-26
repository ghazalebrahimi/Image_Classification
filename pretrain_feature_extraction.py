import tensorflow as tf
import time
import numpy as np
import fetch_data

data_training, data_testing = fetch_data.load_CIFAR_dataset(shuffle=False, percentage=10)

graph_def = tf.GraphDef()
with open('data/inception-2015-12-05/classify_image_graph_def.pb', "rb") as f:
    graph_def.ParseFromString(f.read())
tf.import_graph_def(graph_def, name='')


nsamples_training = len(data_training)
nsamples_testing = len(data_testing)

nsamples = nsamples_training + nsamples_testing

X_data = [data_training[i][0] for i in range(nsamples_training)] + \
         [data_testing[i][0] for i in range(nsamples_testing)]

y_training = np.array([data_training[i][1] for i in range(nsamples_training)])
y_testing = np.array([data_testing[i][1] for i in range(nsamples_testing)])


with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    representation_tensor = sess.graph.get_tensor_by_name('pool_3:0')
    predictions = np.zeros((nsamples, 1008), dtype='float32')
    representations = np.zeros((nsamples, 2048), dtype='float32')

    print('nsamples = ', nsamples)
    start = time.time()
    for i in range(nsamples):
        [reps, preds] = sess.run([representation_tensor, softmax_tensor], {'DecodeJpeg:0': X_data[i]})
        predictions[i] = np.squeeze(preds)
        representations[i] = np.squeeze(reps)
    print('Elapsed time %.1f seconds' % (time.time()-start))


print(predictions[0, :3])
print(representations[0, :3])

np.savez_compressed("data/CIFAR10_features.npz", features_training=representations[:nsamples_training],
                    features_testing=representations[-nsamples_testing:],
                    labels_training=y_training,
                    labels_testing=y_testing)
