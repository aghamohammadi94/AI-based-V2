
## Executable script (optional) ##

# load used libraries
import os
import cv2

import configs.config as config
from ..data.face_cropper import FaceCropper


cropper = FaceCropper()


input_dir = config.RAW_IMAGES_DIR
output_dir = config.ORIGINAL_DATASET_DIR
os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(input_dir):
    img = cv2.imread(os.path.join(input_dir, img_name))
    face = cropper.crop_face(img)
    
    if face is not None:
        cv2.imwrite(os.path.join(output_dir, img_name), face)