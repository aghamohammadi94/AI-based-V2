
## Hijab Detection model testing with webcam

# load used libraries
import numpy as np
import cv2 # for reading images # to read frame by frame images from the webcam
import json # using json to retrieve the information of a file in the form of a dictionary
from keras.models import load_model # to load the trained model # to load the saved model

from .models.conv_base import conv_base
import configs.config as config


def main():
    # Load model
    model = load_model(config.MODEL_PATH)

    # Load class mapping
    with open(config.TRAIN_DICTIONARY, 'r') as file:
        class_indices = json.load(file)

    idx_to_class = {v: k for k, v in class_indices.items()}

    # Open webcam
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        raise RuntimeError('Cannot open webcam')

    # Main loop
    while True:
        ret, frame = webcam.read()
        if not ret:
            break
        
        # Preprocess
        frame_resized = cv2.resize(frame, config.TARGET_SIZE)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_array = frame_rgb / 255.0
        frame_array = np.expand_dims(frame_array, axis=0)

        # Feature extraction
        features = conv_base.predict(frame_array)
        features = features.reshape((1, -1))

        # Prediction
        prediction = model.predict(features)[0][0]
        label = idx_to_class[1] if prediction > 0.5 else idx_to_class[0]


        # Show result
        cv2.putText(
            frame,
            f"Prediction: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (128, 0, 128),
            lineType=cv2.LINE_AA)

        cv2.imshow('Hijab Classification - Webcam', frame)       

        # Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()