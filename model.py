import tensorflow as tf
from tensorflow import keras

def create_model():
    # adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s/net.py
    input_shape = (200, 200, 1)
    inputs = keras.Input(shape=input_shape)

    conv_1a = keras.layers.Conv2D(32, (3, 3), (1, 1), padding='same', activation=tf.nn.relu)(inputs)
    conv_1b = keras.layers.Conv2D(32, (3, 3), (1, 1), padding='same', activation=tf.nn.relu)(conv_1a)
    pool_1 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv_1b)

    conv_2 = keras.layers.Conv2D(64, (3, 3), (1, 1), padding='same', activation=tf.nn.relu)(pool_1)
    pool_2 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv_2)

    conv_3 = keras.layers.Conv2D(128, (3, 3), (1, 1), padding='same', activation=tf.nn.relu)(pool_2)
    pool_3 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv_3)

    fconv_4 = keras.layers.Conv2D(512, (7, 7))(pool_3)
    drop_4 = keras.layers.Dropout(rate=0.5)(fconv_4)

    fconv_5 = keras.layers.Conv2D(512, (1, 1))(drop_4)
    drop_5 = keras.layers.Dropout(rate=0.5)(fconv_5)

    score_fr = keras.layers.Conv2D(21, (1, 1))(drop_5)
    upscore_2 = keras.layers.Conv2DTranspose(21, (4, 4), (2, 2), use_bias=False)(score_fr)

    crop_shape = upscore_2.shape.as_list()[1:3]
    print(crop_shape)
    score_pool_3 = keras.layers.Conv2D(21, (1, 1))(pool_3)
    crop_score_pool_3 = keras.layers.Cropping2D(crop_shape)(score_pool_3)
    fuse_pool_3 = keras.layers.Add()([crop_score_pool_3, score_pool_3])
    upscore_4 = keras.layers.Conv2DTranspose(21, (4, 4), (2, 2), use_bias=False)
    
    upsample_1 = keras.layers.UpSampling2D()(pool_3)
    outputs = keras.layers.Conv2DTranspose()()

    model = keras.Model(input=inputs, output=outputs)
    model.compile(
        optimizer=tf.train.AdamOptimizer(0.001), 
        loss='mean_squared_error',
        metrics=['mean_absolute_error', 'mean_squared_error'])

    return model

if __name__ == "__main__":
    create_model()