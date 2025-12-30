
# load used libraries
import os
import cv2

from ..data.face_cropper import FaceCropper
import configs.config as config


RAW_DIR = config.RAW_IMAGES_DIR
OUTPUT_DIR = config.ORIGINAL_DATASET_DIR                           
                             

LABEL_MAP = {
    'yes': 'hijab',
    'no': 'without_hijab'
}


def main():
    cropper = FaceCropper()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    for img_name in os.listdir(RAW_DIR):
        img_path = os.path.join(RAW_DIR, img_name)
        
        # Image file only
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Identifying labels from file names
        label_key = None
        for key in LABEL_MAP.keys():
            if img_name.lower().startswith(key):
                label_key = key
                break
            
        if label_key is None:
            continue
        
        output_folder = LABEL_MAP[label_key]
        save_dir = os.path.join(OUTPUT_DIR, output_folder)
        os.makedirs(save_dir, exist_ok=True)
        
        face = cropper.crop_face(img)
        if face is None:
            continue
        
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, face)
            


if __name__ == '__main__':
    main()