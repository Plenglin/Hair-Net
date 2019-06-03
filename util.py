import tensorflow as tf

import random
import cv2
import numpy as np
import math


FACELESS_RATE = 0.1
OCCLUDE_RATE = 0.3
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
                select = file_listing.iloc[random.randrange(0, len(file_listing))]
                in_img = cv2.imread(select["input"])
                in_img = cv2.resize(in_img, (224, 224))
                out_img = cv2.imread(select["output"])

                # Random orientations
                rotation = random.choice([None, None, None, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE])
                if rotation is not None:
                    in_img = cv2.rotate(in_img, rotation)
                    out_img = cv2.rotate(out_img, rotation)
                
                # Random occlusions
                if random.random() < OCCLUDE_RATE:
                    points = [np.int0(cv2.boxPoints(((random.randrange(0, 224), random.randrange(0, 224)), (random.randrange(0, 224), random.randrange(0, 224)), random.random() * math.pi)))]
                    color = random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)
                    cv2.drawContours(in_img, points, 0, color, -1)
                    cv2.drawContours(out_img, points, 0, (0, 0, 0), -1)

                # Random scalings
                scale = random.random() / 2 + 0.75
                in_img = cv2.resize(in_img, None, fx=scale, fy=scale)
                out_img = cv2.resize(out_img, None, fx=scale, fy=scale)
                
                # Rescale again to potentially destroy the quality
                scale = random.random() / 2 + 0.75
                in_img = cv2.resize(in_img, None, fx=scale, fy=scale)
                out_img = cv2.resize(out_img, None, fx=scale, fy=scale)

                w, h, _ = in_img.shape
                if w < 224:
                    # Make a border
                    border = (224 - w) // 2
                    in_img = cv2.copyMakeBorder(in_img, border, border, border, border, cv2.BORDER_WRAP)
                    out_img = cv2.copyMakeBorder(out_img, border, border, border, border, cv2.BORDER_WRAP)
                else:
                    # Crop to center
                    size = 224
                    x = random.randint(0, w - size)
                    y = random.randint(0, h - size)
                    in_img = in_img[x:x+size, y:y+size]
                    out_img = out_img[x:x+size, y:y+size]

                yield cv2.resize(in_img, (224, 224)), cv2.resize(out_img, (224, 224))[:,:,1:3]
                
    return tf.data.Dataset.from_generator(_gen, (tf.float32, tf.float32), ((224, 224, 3), (224, 224, 2)))
