import tensorflow as tf

import random
import cv2
import numpy as np


FACELESS_RATE = 0.1
BLANK = np.zeros((224, 224, 2), dtype=np.float)

def _img_map(input_filename, output_filename):
    input_image = tf.read_file(input_filename)
    input_image = tf.image.decode_jpeg(input_image, channels=3)
    input_image = tf.image.resize_images(input_image, (224, 224))
    input_image = tf.cast(input_image, tf.float32)

    output_image = tf.read_file(output_filename)
    output_image = tf.image.decode_bmp(output_image, channels=3)
    output_image = tf.image.resize_images(output_image, (224, 224))
    output_image = tf.cast(output_image, tf.float32)
    output_image = output_image[:, :, 0:2]

    return input_image, output_image


def create_dataset_from_file_listing(file_listing, faceless):
    in_imgs = file_listing
    out_imgs = file_listing
    def _gen():
        while True:
            if random.random() < FACELESS_RATE:
                file = random.choice(faceless)
                img = cv2.imread(file)
                w, h, _ = img.shape
                size = random.randint(122, min(w, h))
                x = random.randint(0, w - size)
                y = random.randint(0, h - size)
                crop = img[x:x+size, y:y+size, :]
                img = cv2.resize(crop, (224, 224))
                yield img, BLANK
            else:
                select = file_listing.iloc[random.randint(0, len(file_listing) - 1)]
                in_img = cv2.imread(select["input"])
                in_img = cv2.resize(in_img, (224, 224))
                out_img = cv2.imread(select["output"])
                out_img = cv2.resize(out_img, (224, 224))
                yield in_img[:,:], out_img[:,:,1:3]
                
    #return tf.data.Dataset.from_tensor_slices(
    #    (file_listing["input"], file_listing["output"])
    #).map(_img_map)

    return tf.data.Dataset.from_generator(_gen, (tf.float32, tf.float32), ((224, 224, 3), (224, 224, 2)))
