
## Hijab Detection model testing on new images


# load used libraries
import json # using json to retrieve the information of a file in the form of a dictionary
from keras.models import load_model # to load the trained model # to load the saved model

from .models.conv_base import conv_base
from artifacts.utils.image_utils import load_and_preprocess_image
import configs.config as config


def main():
    print("Inference started...")
    
    # load the trained model
    model = load_model(config.MODEL_PATH)

    # retrieve values of text file as dictionary variable
    ## with open('./train_dictionary.txt', 'r') as file:
    with open(config.TRAIN_DICTIONARY, 'r') as file:
        class_indices = json.load(file)

    # Reverse the dictionary
    idx_to_class = {v: k for k, v in class_indices.items()}

    img = load_and_preprocess_image(config.TEST_DIR)

    features = conv_base.predict(img)
    features = features.reshape((1, -1))

    prediction = model.predict(features)[0][0]
        
    label = idx_to_class[1] if prediction > 0.5 else idx_to_class[0]
        
    print(f"Prediction result: {label}")
    
    
if __name__ == '__main__':
    main()
