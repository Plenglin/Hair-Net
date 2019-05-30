import tensorflow as tf
from tensorflow import keras


INPUT_SHAPE = (224, 224, 3)
OUTPUT_CHANNELS = 2

def create_model():
    # adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s/net.py

    inputs = keras.Input(shape=INPUT_SHAPE)

    # 224 x 224 x 3
    conv_1 = keras.layers.Conv2D(8, (3, 3), padding='same', activation=tf.nn.relu)(inputs)
    conv_1 = keras.layers.Conv2D(8, (3, 3), padding='same', activation=tf.nn.relu)(conv_1)

    # 112 x 112 x 8
    pool_2 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv_1)
    conv_2 = keras.layers.Conv2D(16, (3, 3), padding='same', activation=tf.nn.relu)(pool_2)
    conv_2 = keras.layers.Conv2D(16, (3, 3), padding='same', activation=tf.nn.relu)(conv_2)

    # 56 x 56 x 16
    pool_3 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv_2)
    conv_3 = keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu)(pool_3)
    conv_3 = keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu)(conv_3)

    # 28
    pool_4 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv_3)
    conv_4 = keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu)(pool_4)
    conv_4 = keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu)(conv_4)
    conv_4 = keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu)(conv_4)

    # 14
    pool_5 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv_4)
    conv_5 = keras.layers.Conv2D(128, (3, 3), padding='same', activation=tf.nn.relu)(pool_5)
    conv_5 = keras.layers.Conv2D(128, (3, 3), padding='same', activation=tf.nn.relu)(conv_5)
    conv_5 = keras.layers.Conv2D(128, (3, 3), padding='same', activation=tf.nn.relu)(conv_5)

    # 7
    pool_6 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv_5)
    fconv_1 = keras.layers.Conv2D(512, (7, 7), padding='same', activation=tf.nn.relu)(pool_6)
    drop_1 = keras.layers.Dropout(rate=0.5)(fconv_1)
    fconv_2 = keras.layers.Conv2D(512, (1, 1), padding='same', activation=tf.nn.relu)(drop_1)
    drop_2 = keras.layers.Dropout(rate=0.5)(fconv_2)
    fconv_3 = keras.layers.Conv2D(64, (1, 1), padding='same', activation=tf.nn.relu)(drop_2)
    drop_3 = keras.layers.Dropout(rate=0.5)(fconv_3)

    # 14
    unpool_5 = keras.layers.UpSampling2D(size=(2, 2))(drop_3)
    fuse_5 = keras.layers.Add()([unpool_5, pool_5])
    deconv_5 = keras.layers.Conv2DTranspose(32, (2, 2), (1, 1), padding='same', activation=tf.nn.relu)(fuse_5)
    deconv_5 = keras.layers.Conv2DTranspose(32, (2, 2), (1, 1), padding='same', activation=tf.nn.relu)(deconv_5)
    deconv_5 = keras.layers.Conv2DTranspose(32, (2, 2), (1, 1), padding='same', activation=tf.nn.relu)(deconv_5)

    # 28
    unpool_4 = keras.layers.UpSampling2D(size=(2, 2))(deconv_5)
    print(unpool_4.shape, pool_4.shape)
    fuse_4 = keras.layers.Add()([unpool_4, pool_4])
    deconv_4 = keras.layers.Conv2DTranspose(16, (2, 2), (1, 1), padding='same', activation=tf.nn.relu)(fuse_4)
    deconv_4 = keras.layers.Conv2DTranspose(16, (2, 2), (1, 1), padding='same', activation=tf.nn.relu)(deconv_4)
    deconv_4 = keras.layers.Conv2DTranspose(16, (2, 2), (1, 1), padding='same', activation=tf.nn.relu)(deconv_4)

    # 56
    unpool_3 = keras.layers.UpSampling2D(size=(2, 2))(deconv_4)
    fuse_3 = keras.layers.Add()([unpool_3, pool_3])
    deconv_3 = keras.layers.Conv2DTranspose(8, (2, 2), (1, 1), padding='same', activation=tf.nn.relu)(fuse_3)
    deconv_3 = keras.layers.Conv2DTranspose(8, (2, 2), (1, 1), padding='same', activation=tf.nn.relu)(deconv_3)

    # 112
    unpool_2 = keras.layers.UpSampling2D(size=(2, 2))(deconv_3)
    fuse_2 = keras.layers.Add()([unpool_2, pool_2])
    deconv_2 = keras.layers.Conv2DTranspose(8, (2, 2), (1, 1), padding='same', activation=tf.nn.relu)(fuse_2)
    deconv_2 = keras.layers.Conv2DTranspose(8, (2, 2), (1, 1), padding='same', activation=tf.nn.relu)(deconv_2)

    # 224
    unpool_1 = keras.layers.UpSampling2D(size=(2, 2))(deconv_2)
    fuse_1 = keras.layers.Add()([unpool_1, conv_1])
    deconv_1 = keras.layers.Conv2DTranspose(4, (2, 2), (1, 1), padding='same', activation=tf.nn.relu)(fuse_1)
    deconv_1 = keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, (2, 2), (1, 1), padding='same', activation=tf.nn.relu)(deconv_1)

    model = keras.Model(inputs=inputs, outputs=deconv_1)
    model.compile(
        optimizer=tf.train.AdamOptimizer(0.001), 
        loss='mean_squared_error',
        metrics=['mean_absolute_error', 'mean_squared_error'])

    return model


if __name__ == "__main__":
    create_model()
