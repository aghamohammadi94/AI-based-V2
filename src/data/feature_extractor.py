
## Feature Extractor
## Here VGG16 pre-trained network was used for feature extraction

# load used libraries
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from ..models.conv_base import conv_base
import configs.config as config


# creating a function for feature extraction with VGG16 pre-trained network
def extract_features(directory, sample_count):
    
    # by dividing the matrix of images by 255, we change the value of the matrix to the range of 0 to 1
    datagen = ImageDataGenerator(rescale=1./255)

    ## batch_size = 20
    batch_size = config.BATH_SIZE
    
    # by changing the input size to (150, 150, 3) and after removing all fully connected layers
    # the size of the last layer of the VGG16 pre-trained network is changed to (4, 4, 512)
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    
    generator = datagen.flow_from_directory(
        directory,
        ## target_size=(150, 150),
        target_size=config.TARGET_SIZE,
        batch_size=batch_size,
        class_mode='binary')
    
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
        
    return features, labels, generator.class_indices

