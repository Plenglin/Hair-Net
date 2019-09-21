import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

import time

import util



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
graph = tf.Graph()
with tf.Session(graph=graph, config=config) as sess:
    tf.keras.backend.set_session(sess)
    hairnet_def = tf.contrib.saved_model.load_keras_model("./saved_models/1559980328")

    #print(graph.get_operations())
    input_layer = graph.get_tensor_by_name("input:0")
    output_layer = graph.get_tensor_by_name("output/Sigmoid:0")

    hair_mask = np.zeros((224, 224, 3), dtype=np.float)
    face_mask = np.zeros((224, 224, 3), dtype=np.float)
    hair_mask[:, :, 0] = 255
    face_mask[:, :, 1] = 255
    cam = cv2.VideoCapture(0)
    while cv2.waitKey(1) & 0xFF != ord('q'):
        ret, frame = cam.read()
        width, height, _ = frame.shape
        dim = min(width, height)
        ox = (width - dim) // 2
        oy = (height - dim) // 2
        frame = frame[ox:ox+dim, oy:oy+dim]
        frame = cv2.resize(frame, (224, 224))

        start = time.time()
        result = sess.run(output_layer, feed_dict={"input:0": [frame]})
        print(f"FPS: {1 / (time.time() - start)}")
        output_face = np.reshape(np.repeat(result[0, :, :, 1], 3, axis=2), (224, 224, 3))
        output_hair = result[0, :, :, 2]

        frame = frame * (1 - output_face) + face_mask * output_face
        
        cv2.imshow('img', frame)
