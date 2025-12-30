
## Hijab Detection model

# load used libraries
import os
import numpy as np
import json # to save variable values ​​in a text file
from tensorflow.keras.applications.vgg16 import VGG16 # to download VGG16 pre-trained network
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import optimizers

from .data.feature_extractor import extract_features
from artifacts.utils.plot_history import plot_training
import configs.config as config


def build_model(num_classes=1):
    ## here VGG16 pre-trained network was used for feature extraction

    # download the VGG16 pre-trained network without fully connected layers and without Softmax layer
    # by changing the input size to (150, 150, 3) and after removing all fully connected layers
    # the size of the last layer of the VGG16 pre-trained network is changed to (4, 4, 512)
    conv_base = VGG16(weights='imagenet',
                    include_top=False,
                    ## input_shape=(150, 150, 3))
                    input_shape=config.INPUT_SHAPE)
    
    # by setting the value False for conv_base.trainable when training a new model, we will not train the weights of the VGG16 network
    for layer in conv_base.layers:
        layer.trainable = False
       
    # making a new model and combining it with the VGG16 model
    # here VGG16 pre-trained network was used for feature extraction 
    x = layers.Flatten()(conv_base.output)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(num_classes, activation='sigmoid')(x)
    
    return Model(inputs=conv_base.input, outputs=output)


def main():
    
    # main dataset directory path
    ## dataset_dir = './datasets'
    dataset_dir = config.DATASET_DIR

    # directory for training
    train_dir = os.path.join(dataset_dir, 'train')

    # directory for validation
    validation_dir = os.path.join(dataset_dir, 'validation')
    
    
    ## train_features, train_labels, train_dictionary = extract_features(train_dir, 2000)
    train_features, train_labels, train_dictionary = extract_features(train_dir, config.NUMBER_TRAINING_IMAGES)

    train_features = np.reshape(train_features, (-1, 4 * 4 * 512))


    # save the train_dictionary variable values ​​to a text file
    ## with open('train_dictionary.txt', 'w') as file:
    with open(config.TRAIN_DICTIONARY, 'w') as file:
        json.dump(train_dictionary, file)
        
        
    ## batch_size = 20
    batch_size = config.BATH_SIZE
    
    ## Data Augmentation
    # augment the training data
    # since the number of images to train the model is small, data augmentation was used to augment the training data

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # the validation data should not be augmented
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            # this is the target directory
            train_dir,
            # all images will be resized to 150x150
            ## target_size=(150, 150),
            target_size=config.TARGET_SIZE,
            batch_size=batch_size,
            # since we use binary_crossentropy loss, we need binary labels
            class_mode='binary',
            subset='training')

    validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            ## target_size=(150, 150),
            target_size=config.TARGET_SIZE,
            batch_size=batch_size,
            class_mode='binary',
            subset='validation')
    

    model = build_model(num_classes=1)
        
    
    model.compile(loss='binary_crossentropy',
                ## optimizer=optimizers.RMSprop(learning_rate=2e-5),
                optimizer=optimizers.RMSprop(learning_rate=config.LEARNING_RATE),
                metrics=['acc'])

    # Model.fit_generator is deprecated
    history = model.fit(
        train_generator,
        ## steps_per_epoch=100,
        steps_per_epoch=config.STEP_PER_EPOCH,
        ## epochs=40,
        epochs=config.EPOCHS,
        validation_data=validation_generator,
        ## validation_steps=30)
        validation_steps=config.VALIDATION_STEPS)


    model.save('./Hijab_Detection_model_with_vgg16.h5')

    plot_training(history, config.PLOTS_DIR)
    
    loss, acc = model.evaluate(train_features, train_labels)
    print(f"Training accuracy: {acc:.4f}")
    print(f"Training loss: {loss:.4f}")
    
     
if __name__ == '__main__':
    main()

