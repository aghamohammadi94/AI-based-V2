
# load used libraries
import numpy as np
from keras.utils import load_img
from keras.utils import img_to_array

import configs.config as config


def load_and_preprocess_image(img_path):
    
    img = load_img(img_path, target_size=config.TARGET_SIZE)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array
