
# load used libraries
from tensorflow.keras.applications.vgg16 import VGG16

import configs.config as config


# download the VGG16 pre-trained network without fully connected layers and without Softmax layer
# by changing the input size to (150, 150, 3) and after removing all fully connected layers
# the size of the last layer of the VGG16 pre-trained network is changed to (4, 4, 512)
conv_base = VGG16(weights='imagenet',
                include_top=False,
                ## input_shape=(150, 150, 3))
                input_shape=config.INPUT_SHAPE)

