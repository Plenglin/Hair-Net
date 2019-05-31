import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from tensorboard.plugins.beholder import Beholder

import model
import torchvision


LOG_DIR = './logs'
EPOCHS = 1000
STEPS_PER_EPOCH = 100
BATCH_SIZE = 10

file_listing = pd.read_csv('train.csv')

output_transformer_resize = torchvision.transforms.Resize((224, 224))
output_transformer_tensor = torchvision.transforms.ToTensor()

def _img_map(input_filename, output_filename):
    input_image = tf.read_file(input_filename)
    input_image = tf.image.decode_jpeg(input_image, channels=3)
    input_image = tf.image.resize_images(input_image, (224, 224))
    input_image = tf.cast(input_image, tf.float32)

    output_image = tf.read_file(output_filename)
    output_image = tf.image.decode_bmp(output_image, channels=3)
    output_image = tf.image.resize_images(output_image, (224, 224))
    output_image = tf.cast(output_image, tf.float32)
    output_image = output_image[:,:,0:1]

    return input_image, output_image

dataset = tf.data.Dataset.from_tensor_slices((file_listing['input'], file_listing['output']))
dataset = dataset.map(_img_map)
dataset = dataset.batch(BATCH_SIZE)
iterator = dataset.repeat().make_one_shot_iterator()
images, labels = iterator.get_next()


with tf.Session() as sess:
    tf.keras.backend.set_session(sess)
    hairnet = model.create_model()
    beholder = Beholder(logdir=LOG_DIR)

    def beholder_on_epoch_begin(x, y):
        beholder.update(
            session=sess,
            frame=hairnet.get_layer('output').output
        )
    beholder_callback = tf.keras.callbacks.LambdaCallback(on_epoch_begin=beholder_on_epoch_begin)
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

