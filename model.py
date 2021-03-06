import tensorflow as tf
from tensorflow import keras


INPUT_SHAPE = (224, 224, 3)
OUTPUT_CHANNELS = 2


def create_model():
    # adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s/net.py

    inputs = keras.Input(shape=INPUT_SHAPE, name="input")

    # 224 x 224 x 3
    conv_1 = keras.layers.Conv2D(
        8, (3, 3), padding="same", activation=tf.nn.relu, name="conv_1a"
    )(inputs)
    conv_1 = keras.layers.Conv2D(
        8, (3, 3), padding="same", activation=tf.nn.relu, name="conv_1b"
    )(conv_1)

    # 112 x 112 x 8
    pool_2 = keras.layers.MaxPool2D(pool_size=(2, 2), name="pool_2")(conv_1)
    conv_2 = keras.layers.Conv2D(
        16, (3, 3), padding="same", activation=tf.nn.relu, name="conv_2a"
    )(pool_2)
    conv_2 = keras.layers.Conv2D(
        16, (3, 3), padding="same", activation=tf.nn.relu, name="conv_2b"
    )(conv_2)

    # 56 x 56 x 16
    pool_3 = keras.layers.MaxPool2D(pool_size=(2, 2), name="pool_3")(conv_2)
    conv_3 = keras.layers.Conv2D(
        32, (3, 3), padding="same", activation=tf.nn.relu, name="conv_3a"
    )(pool_3)
    conv_3 = keras.layers.Conv2D(
        32, (3, 3), padding="same", activation=tf.nn.relu, name="conv_3b"
    )(conv_3)

    # 28
    pool_4 = keras.layers.MaxPool2D(pool_size=(2, 2), name="pool_4")(conv_3)
    conv_4 = keras.layers.Conv2D(
        64, (3, 3), padding="same", activation=tf.nn.relu, name="conv_4a"
    )(pool_4)
    conv_4 = keras.layers.Conv2D(
        64, (3, 3), padding="same", activation=tf.nn.relu, name="conv_4b"
    )(conv_4)
    conv_4 = keras.layers.Conv2D(
        64, (3, 3), padding="same", activation=tf.nn.relu, name="conv_4c"
    )(conv_4)

    # 14
    pool_5 = keras.layers.MaxPool2D(pool_size=(2, 2), name="pool_5")(conv_4)
    conv_5 = keras.layers.Conv2D(
        128, (3, 3), padding="same", activation=tf.nn.relu, name="conv_5a"
    )(pool_5)
    conv_5 = keras.layers.Conv2D(
        128, (3, 3), padding="same", activation=tf.nn.relu, name="conv_5b"
    )(conv_5)
    conv_5 = keras.layers.Conv2D(
        128, (3, 3), padding="same", activation=tf.nn.relu, name="conv_5c"
    )(conv_5)

    # 7
    pool_6 = keras.layers.MaxPool2D(pool_size=(2, 2), name="pool_6")(conv_5)
    fconv_1 = keras.layers.Conv2D(
        512, (5, 5), padding="same", activation=tf.nn.relu, name="fc_1"
    )(pool_6)
    drop_1 = keras.layers.Dropout(rate=0.5)(fconv_1)
    fconv_2 = keras.layers.Conv2D(
        512, (1, 1), padding="same", activation=tf.nn.relu, name="fc_2"
    )(drop_1)
    drop_2 = keras.layers.Dropout(rate=0.5)(fconv_2)

    score_6 = keras.layers.Conv2D(
        256, (3, 3), padding="same", activation=tf.nn.relu, name="score_6"
    )(pool_6)
    deconv_6 = keras.layers.Conv2DTranspose(
        256, (1, 1), padding="same", activation=tf.nn.relu, name="deconv_6"
    )(drop_2)
    fuse_6 = keras.layers.Add()([deconv_6, score_6])

    # 14
    score_5 = keras.layers.Conv2D(
        256, (3, 3), (1, 1), padding="same", activation=tf.nn.relu, name="score_5"
    )(pool_5)
    unpool_5 = keras.layers.UpSampling2D(size=(2, 2), name="unpool_5")(fuse_6)
    fuse_5 = keras.layers.Add()([unpool_5, score_5])
    deconv_5 = keras.layers.Conv2DTranspose(
        32, (3, 3), (1, 1), padding="same", activation=tf.nn.relu, name="deconv_5a"
    )(fuse_5)

    # 28
    score_4 = keras.layers.Conv2D(
        32, (3, 3), (1, 1), padding="same", activation=tf.nn.relu, name="score_4"
    )(pool_4)
    unpool_4 = keras.layers.UpSampling2D(size=(2, 2), name="unpool_4")(deconv_5)
    fuse_4 = keras.layers.Add()([unpool_4, score_4])
    deconv_4 = keras.layers.Conv2DTranspose(
        16, (3, 3), (1, 1), padding="same", activation=tf.nn.relu, name="deconv_4a"
    )(fuse_4)

    # 56
    score_3 = keras.layers.Conv2D(
        16, (3, 3), (1, 1), padding="same", activation=tf.nn.relu, name="score_3"
    )(pool_3)
    unpool_3 = keras.layers.UpSampling2D(size=(2, 2), name="unpool_3")(deconv_4)
    fuse_3 = keras.layers.Add()([unpool_3, score_3])
    deconv_3 = keras.layers.Conv2DTranspose(
        8, (3, 3), (1, 1), padding="same", activation=tf.nn.relu, name="deconv_3a"
    )(fuse_3)
    deconv_3 = keras.layers.Conv2DTranspose(
        8, (3, 3), (1, 1), padding="same", activation=tf.nn.relu, name="deconv_3b"
    )(deconv_3)

    # 112
    score_2 = keras.layers.Conv2D(
        8, (3, 3), (1, 1), padding="same", activation=tf.nn.relu, name="score_2"
    )(pool_2)
    unpool_2 = keras.layers.UpSampling2D(size=(2, 2), name="unpool_2")(deconv_3)
    fuse_2 = keras.layers.Add()([unpool_2, score_2])
    deconv_2 = keras.layers.Conv2DTranspose(
        8, (5, 5), (1, 1), padding="same", activation=tf.nn.relu, name="deconv_2a"
    )(fuse_2)
    deconv_2 = keras.layers.Conv2DTranspose(
        8, (5, 5), (1, 1), padding="same", activation=tf.nn.relu, name="deconv_2c"
    )(deconv_2)

    # 224
    score_1 = keras.layers.Conv2D(
        8, (3, 3), (1, 1), padding="same", activation=tf.nn.relu, name="score_1"
    )(inputs)
    unpool_1 = keras.layers.UpSampling2D(size=(2, 2), name="unpool_1")(deconv_2)
    fuse_1 = keras.layers.Add()([unpool_1, score_1])
    deconv_1 = keras.layers.Conv2DTranspose(
        4, (5, 5), padding="same", activation=tf.nn.relu, name="deconv_1a"
    )(fuse_1)
    deconv_1 = keras.layers.Conv2DTranspose(
        4, (5, 5), padding="same", activation=tf.nn.relu, name="deconv_1b"
    )(deconv_1)
    deconv_1 = keras.layers.Conv2DTranspose(
        OUTPUT_CHANNELS, (5, 5), padding="same", activation=tf.nn.relu, name="output"
    )(deconv_1)

    model = keras.Model(inputs=inputs, outputs=deconv_1)

    return model


if __name__ == "__main__":
    create_model()
