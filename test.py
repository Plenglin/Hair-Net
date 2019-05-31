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
    image_la = tf.read_file(file_listing.iloc[7]['output'])
    image_la = tf.image.decode_bmp(image_la, channels=3)
    image_la = tf.image.resize_images(image_la, (224, 224))
    #image_la = tf.cast(image_la, tf.float32)
    image_la = sess.run(image_la[:,:,0:2])

    input_layer = graph.get_tensor_by_name('input:0')
    output_layer = graph.get_tensor_by_name('output/Relu:0')
    print(output_layer)

    result = sess.run(output_layer, feed_dict={'input:0': [image_in]})
    channel_a = result[0,:,:,0]
    channel_b = result[0,:,:,1]
    plt.subplot(3, 2, 1)
    plt.imshow(image_in)
    plt.subplot(3, 2, 2)
    plt.imshow(channel_a)
    plt.subplot(3, 2, 3)
    plt.imshow(channel_b)
    plt.subplot(3, 2, 4)
    plt.imshow(image_la[:,:,0])
    plt.subplot(3, 2, 5)
    plt.imshow(image_la[:,:,1])
    plt.show()
