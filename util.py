import tensorflow as tf


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


def create_dataset_from_file_listing(file_listing):
    return tf.data.Dataset.from_tensor_slices(
        (file_listing["input"], file_listing["output"])
    ).map(_img_map)
