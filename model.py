import tensorflow as tf
from tensorflow import keras


INPUT_SHAPE = (224, 224, 3)

def create_model(classes=2):
    # adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s/net.py

    inputs = keras.Input(shape=INPUT_SHAPE, name="input")

    # 112 x 112 x 8
    conv_2 = keras.layers.Conv2D(16, 3, 2, padding="same", activation=tf.nn.relu, name="conv_2a")(inputs)
    conv_2 = keras.layers.Conv2D(16, 3, 1, padding="same", activation=tf.nn.relu, name="conv_2b")(conv_2)

    # 56 x 56 x 16
    conv_3 = keras.layers.Conv2D(64, 3, 2, padding="same", activation=tf.nn.relu, name="conv_3a")(conv_2)
    conv_3 = keras.layers.Conv2D(64, 3, 1, padding="same", activation=tf.nn.relu, name="conv_3b")(conv_3)

    # 28
    conv_4 = keras.layers.Conv2D(128, 3, 2, padding="same", activation=tf.nn.relu, name="conv_4a")(conv_3)
    conv_4 = keras.layers.Conv2D(128, 3, 1, padding="same", activation=tf.nn.relu, name="conv_4b")(conv_4)
    conv_4 = keras.layers.Conv2D(128, 3, 1, padding="same", activation=tf.nn.relu, name="conv_4c")(conv_4)

    # 14
    conv_5 = keras.layers.Conv2D(128, 3, 2, padding="same", activation=tf.nn.relu, name="conv_5a")(conv_4)
    conv_5 = keras.layers.Conv2D(128, 3, 1, padding="same", activation=tf.nn.relu, name="conv_5b")(conv_5)
    conv_5 = keras.layers.Conv2D(128, 3, 1, padding="same", activation=tf.nn.relu, name="conv_5c")(conv_5)

    # 7
    fconv_1 = keras.layers.Conv2D(512, 3, 2, padding="same", activation=tf.nn.relu, name="fc_1")(conv_5)
    drop_1 = keras.layers.Dropout(rate=0.5)(fconv_1)
    fconv_2 = keras.layers.Conv2D(512, 1, 1, padding="same", activation=tf.nn.relu, name="fc_2")(drop_1)
    drop_2 = keras.layers.Dropout(rate=0.5)(fconv_2)
    score_6 = keras.layers.Conv2D(512, 1, 1, padding="same", activation=tf.nn.relu, name="fc_3")(drop_2)
    drop_3 = keras.layers.Dropout(rate=0.5)(score_6)

    # 14
    score_5 = keras.layers.Conv2D(64, 1, 1, padding="same", activation=tf.nn.relu, name="score_5")(conv_5)
    deconv_5 = keras.layers.Conv2DTranspose(64, 1, 2, padding="same", activation=tf.nn.relu, name="deconv_5")(drop_3)
    fuse_5 = keras.layers.Add()([deconv_5, score_5])

    # 28
    score_4 = keras.layers.Conv2D(32, 1, 1, padding="same", activation=tf.nn.relu, name="score_4")(conv_4)
    deconv_4 = keras.layers.Conv2DTranspose(32, 3, 2, padding="same", activation=tf.nn.relu, name="deconv_4")(fuse_5)
    fuse_4 = keras.layers.Add()([deconv_4, score_4])

    # 56
    score_3 = keras.layers.Conv2D(32, 1, 1, padding="same", activation=tf.nn.relu, name="score_3")(conv_3)
    deconv_3 = keras.layers.Conv2DTranspose(32, 3, 2, padding="same", activation=tf.nn.relu, name="deconv_3")(fuse_4)
    fuse_3 = keras.layers.Add()([deconv_3, score_3])

    # 112
    score_2 = keras.layers.Conv2D(32, 1, 1, padding="same", activation=tf.nn.relu, name="score_2")(conv_2)
    deconv_2 = keras.layers.Conv2DTranspose(32, 3, 2, padding="same", activation=tf.nn.relu, name="deconv_2")(fuse_3)
    fuse_2 = keras.layers.Add()([deconv_2, score_2])

    # 224
    deconv_1 = keras.layers.Conv2DTranspose(classes, 3, 2, padding="same", activation=tf.nn.relu, name="deconv_1")(fuse_2)
    output = keras.layers.Activation('linear', name='output')(deconv_1)

    model = keras.Model(inputs=inputs, outputs=output)

    return model


if __name__ == "__main__":
    create_model()
