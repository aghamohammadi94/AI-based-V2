
# load used library
import os
import numpy as np
from keras.models import load_model # to load the trained model # to load the saved model

from .data.feature_extractor import extract_features
import configs.config as config


def main():
    
    # load the trained model
    model = load_model(config.MODEL_PATH)
    
    # main dataset directory path
    ## dataset_dir = './datasets'
    dataset_dir = config.DATASET_DIR

    # directory for validation
    validation_dir = os.path.join(dataset_dir, 'validation')
    
    ## validation_features, validation_labels, validation_dictionary = extract_features(validation_dir, 600)
    validation_features, validation_labels, validation_dictionary = extract_features(validation_dir, config.NUMBER_VALIDATION_IMAGES)

    validation_features = np.reshape(validation_features, (-1, 4 * 4 * 512))

    loss, acc = model.evaluate(validation_features, validation_labels)
    
    print(f"Validation accuracy: {acc:.4f}")
    print(f"Validation loss: {loss:.4f}")
    

if __name__ == '__main__':
    main()
