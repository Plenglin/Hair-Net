import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

import time

import util


file_listing = pd.read_csv("test.csv")


graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    tf.keras.backend.set_session(sess)
    hairnet_def = tf.contrib.saved_model.load_keras_model("./saved_models/1559294377")

    index = 120

    image_in = cv2.resize(cv2.imread(file_listing.iloc[index]["input"]), (224, 224))
    image_la = tf.read_file(file_listing.iloc[index]["output"])
    image_la = tf.image.decode_bmp(image_la, channels=3)
    image_la = tf.image.resize_images(image_la, (224, 224))
    # image_la = tf.cast(image_la, tf.float32)
    image_la = sess.run(image_la[:, :, 0:2])
    gtruth = np.zeros_like(image_in)
    gtruth[:, :, 0:2] = image_la

    input_layer = graph.get_tensor_by_name("input:0")
    output_layer = graph.get_tensor_by_name("output/Relu:0")
    print(output_layer)

    start = time.time()
    result = sess.run(output_layer, feed_dict={"input:0": [image_in]})
    print(f"Processed in {time.time() - start} s")
    output_hair = result[0, :, :, 0]
    output_face = result[0, :, :, 1]

    thresh_hair = (output_hair > 50).astype(np.uint8)
    single_channel_mask_hair = thresh_hair * 255
    mask_hair = np.zeros_like(image_in)
    mask_hair[:, :, 0] = single_channel_mask_hair
    mask_hair[:, :, 1] = single_channel_mask_hair
    mask_hair[:, :, 2] = single_channel_mask_hair
    hair_img = cv2.bitwise_and(image_in, mask_hair)

    thresh_face = (output_face > 50).astype(np.uint8) * 255
    mask_face = np.zeros_like(image_in)
    mask_face[:, :, 0] = thresh_face
    mask_face[:, :, 1] = thresh_face
    mask_face[:, :, 2] = thresh_face
    face_img = cv2.bitwise_and(image_in, mask_face)

    face_colors = image_in[output_face > 50, :]
    print(face_colors.shape)
    med_skin_color = np.average(image_in[output_face > 50, :], 0) / 255
    rpl_hair = (mask_hair * med_skin_color).astype(np.uint8)
    print(rpl_hair)
    pseudo_balidfy = cv2.bitwise_and(image_in, cv2.bitwise_not(mask_hair))
    hair_refill = cv2.bitwise_or(pseudo_balidfy, rpl_hair)

    plt.subplot(2, 3, 1)
    plt.title("Original")
    plt.imshow(image_in[:, :, ::-1])
    plt.subplot(2, 3, 4)
    plt.title("Ground Truth")
    plt.imshow(gtruth)
    plt.subplot(2, 3, 2)
    plt.title("Separated Hair")
    plt.imshow(hair_img[:, :, ::-1])
    plt.subplot(2, 3, 3)
    plt.title("Separated Face")
    plt.imshow(face_img[:, :, ::-1])
    plt.subplot(2, 3, 5)
    plt.title("rm -f hair")
    plt.imshow(pseudo_balidfy[:, :, ::-1])
    plt.subplot(2, 3, 6)
    plt.title("Pseudo-Baldify")
    plt.imshow(hair_refill[:, :, ::-1])
    plt.show()
