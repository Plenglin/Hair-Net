import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from tensorboard.plugins.beholder import Beholder

import model
import util


LOG_DIR = './logs'
EPOCHS = 1000
STEPS_PER_EPOCH = 100
BATCH_SIZE = 10

file_listing = pd.read_csv('train.csv')

dataset = util.create_dataset_from_file_listing(file_listing)
iterator = dataset.batch(BATCH_SIZE).repeat().make_one_shot_iterator()
images, labels = iterator.get_next()


with tf.Session() as sess:
    tf.keras.backend.set_session(sess)
    hairnet = model.create_model()
    beholder = Beholder(logdir=LOG_DIR)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss', 
        patience=50
    )
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        'training/cp-{epoch:04d}.ckpt', 
        save_weights_only=True, 
        verbose=1, 
        period=10
    )
    tboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=LOG_DIR,
        batch_size=32, 
        write_graph=True,
        update_freq='epoch'
    )

    hairnet.fit(
        tf.image.resize_images(images, (224, 224)), 
        labels,
        epochs=EPOCHS, 
        #validation_split=0.2,
        steps_per_epoch=100,
        callbacks=[early_stop, cp_callback, tboard_callback])
    
    try:
        os.makedirs('./saved_models')
    except FileExistsError:
        pass
    tf.contrib.saved_model.save_keras_model(hairnet, './saved_models')

