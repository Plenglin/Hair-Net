import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

import model
import util
import datetime


LOG_DIR = "./logs/train_" + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
EPOCHS = 1000
STEPS_PER_EPOCH = 100
BATCH_SIZE = 10

file_listing = pd.read_csv("train.csv")
with open('train_faceless.txt', 'r') as f:
    facelesses = [l[:-1] for l in f.readlines()]

gen = lambda: util.create_gen_from_file_listing(file_listing, facelesses)
dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32), ((224, 224, 3), (224, 224, 3)))
iterator = (dataset
    .batch(BATCH_SIZE)
    .prefetch(8)
    .repeat()
    .make_one_shot_iterator())
images, labels = iterator.get_next()


with tf.Session() as sess:
    tf.keras.backend.set_session(sess)
    hairnet = model.create_model()
    #hairnet = tf.contrib.saved_model.load_keras_model('./saved_models/1559969780')
    run_options = tf.RunOptions()
    run_metadata = tf.RunMetadata()

    hairnet.compile(
        optimizer=tf.train.AdamOptimizer(0.001),
        loss="mean_squared_error",
        metrics=["accuracy", "categorical_crossentropy", "mean_squared_error"],
        options=run_options, 
        run_metadata=run_metadata
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=50)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        "training/cp-{epoch:04d}.ckpt", save_weights_only=True, verbose=1, period=10
    )
    tboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=LOG_DIR, batch_size=32, write_graph=True, update_freq="epoch"
    )

    try:
        hairnet.fit(
            images,
            labels,
            epochs=EPOCHS,
            steps_per_epoch=100,
            callbacks=[early_stop, cp_callback, tboard_callback],
        )
    except KeyboardInterrupt:
        print('KeyboardInterrupt. stopping')

    try:
        os.makedirs("./saved_models")
    except FileExistsError:
        pass
    tf.contrib.saved_model.save_keras_model(hairnet, "./saved_models")
