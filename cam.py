import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

import time

import util



graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    tf.keras.backend.set_session(sess)
    hairnet_def = tf.contrib.saved_model.load_keras_model("./saved_models/1559294377")

    input_layer = graph.get_tensor_by_name("input:0")
    output_layer = graph.get_tensor_by_name("output/Relu:0")

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
        result = sess.run(output_layer, feed_dict={"input:0": [frame[:, :, ::-1]]})
        print(f"FPS: {1 / (time.time() - start)}")
        output_hair = result[0, :, :, 0]
        output_face = result[0, :, :, 1]
    
        frame[output_hair > 50] = (255, 0, 0)
        frame[output_face > 50] = (0, 255, 0)
        cv2.imshow('img', frame)
