import tensorflow as tf
from datasets.base_dataset import DatasetBase

class ImageDatasetBase(DatasetBase):
    def __init__(self, batch_size, img_height, img_width):
        # sets the image dimensions and batch size
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size

    def decode_img(self, image_string, channels, augment=False):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_png(image_string, channels=channels)
        # augment the image
        if augment:
            img = self.apply_img_augmentation(img)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [self.img_height, self.img_width])

    def apply_img_augmentation(self, image):
        image = tf.image.random_saturation(image, 0.5, 1)
        image = tf.image.random_hue(image, 0.08)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1)
        return image
