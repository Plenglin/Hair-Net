import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import util


file_listing = pd.read_csv('test.csv')


graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    tf.keras.backend.set_session(sess)
    hairnet_def = tf.saved_model.loader.load(sess, ['eval'], './saved_models/1559286147')

    image_in = cv2.resize(cv2.imread(file_listing.iloc[7]['input']), (224, 224))

    print([n.name for n in tf.get_default_graph().as_graph_def().node])
    input_layer = graph.get_tensor_by_name('input:0')
    output_layer = graph.get_tensor_by_name('output/Relu:0')
    print(output_layer)

    result = sess.run(output_layer, feed_dict={'input:0': [image_in]})
    channel_a = result[0,:,:,0]
    channel_b = result[0,:,:,1]
    plt.subplot(1, 3, 1)
    plt.imshow(image_in)
    plt.subplot(1, 3, 2)
    plt.imshow(channel_a)
    plt.subplot(1, 3, 3)
    plt.imshow(channel_b)
    plt.show()
