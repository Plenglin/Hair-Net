import tensorflow as tf

import random
import cv2
import numpy as np
import math


FACELESS_RATE = 0.4
OCCLUDE_RATE = 0.3
BLANK = np.zeros((224, 224, 3), dtype=np.float)
BLANK[:, :, 0] = 1.0

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


def create_gen_from_file_listing(file_listing, faceless):
    in_imgs = file_listing
    out_imgs = file_listing
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
            in_file = select["input"]
            in_img = cv2.imread(in_file, flags=cv2.IMREAD_COLOR)
            out_img = cv2.imread(select["output"])
            yield cv2.resize(in_img, (224, 224)), cv2.resize(out_img, (224, 224))

            # Random orientations
            #rotation = random.choice([None, None, None, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE])
            #if rotation is not None:
            #    in_img = cv2.rotate(in_img, rotation)
            #    out_img = cv2.rotate(out_img, rotation)
            
            # Random occlusions
            if random.random() < OCCLUDE_RATE:
                points = [np.int0(cv2.boxPoints(((random.randrange(0, 224), random.randrange(0, 224)), (random.randrange(0, 100), random.randrange(0, 100)), random.random() * math.pi)))]
                color = random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)
                cv2.drawContours(in_img, points, 0, color, -1)
                cv2.drawContours(out_img, points, 0, (0, 0, 0), -1)

            # Random scaling and translation
            scale = random.random() / 2 + 0.75
            tx = (224 - scale * 250) / 2 + (random.random() - 0.5) * 224 / scale
            ty = (224 - scale * 250) / 2 + (random.random() - 0.5) * 224 / scale
            M = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float)
            in_img = cv2.warpAffine(in_img, M, (224, 224))
            out_img = cv2.warpAffine(out_img, M, (224, 224))

            yield in_img, out_img


if __name__ == "__main__":
    import util
    import cv2
    import pandas as pd


    file_listing = pd.read_csv("train.csv")
    with open('train_faceless.txt', 'r') as f:
        facelesses = [l[:-1] for l in f.readlines()]

    gen = util.create_gen_from_file_listing(file_listing, facelesses)

    for i, l in gen:
        a = l[:, :, 0]
        b = l[:, :, 1]

        cv2.imshow('i', i)
        cv2.imshow('a', a)
        cv2.imshow('b', b)
        cv2.waitKey()

