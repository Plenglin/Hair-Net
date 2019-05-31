import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

import model
import torchvision

EPOCHS = 1000
BATCH_SIZE = 10

file_listing = pd.read_csv('data.csv')
X_train, X_test, y_train, y_test = train_test_split(file_listing['input'], file_listing['output'], shuffle=True)

output_transformer_resize = torchvision.transforms.Resize((224, 224))
output_transformer_tensor = torchvision.transforms.ToTensor()

def _gen():
    for input_filename, output_filename in zip(X_train, y_train):
        input_str = tf.read_file(input_filename)
        input_decoded = tf.image.decode_jpeg(input_str, channels=3)
        input_resized = tf.image.resize_images(input_decoded, (224, 224))
        input_raw = tf.cast(input_resized, tf.float32)

        output_raw = Image.open(output_filename)
        output_scaled = output_transformer_resize(output_raw)
        output_hair, output_skin, _ = output_scaled.split()
        
        yield input_resized, (output_transformer_tensor(output_hair), output_transformer_tensor(output_skin))

dataset = tf.data.Dataset.from_generator(
    _gen, 
    (tf.float32, tf.float32),
    ((224, 224, 3), (224, 224, 2))
)
dataset = dataset.batch(BATCH_SIZE)
iterator = dataset.make_one_shot_iterator()
images, labels = iterator.get_next()


with tf.Session() as sess:
    tf.keras.backend.set_session(sess)
    hairnet = model.create_model()

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
    hairnet.fit(
        tf.image.resize_images(images, (224, 224)), labels,
        epochs=EPOCHS, 
        #validation_split=0.2,
        steps_per_epoch=1000,
        callbacks=[early_stop])
    
    try:
        os.makedirs('./saved_models')
    except FileExistsError:
        pass
    tf.contrib.saved_model.save_keras_model(hairnet, './saved_models')

