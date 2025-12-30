
# load used libraries
import os
import shutil
import random

def split_dataset(
    source_dir,
    train_dir,
    val_dir,
    split_ratio = 0.8
):
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    images = os.listdir(source_dir)
    random.shuffle(images)
    
    split_idx = int(len(images) * split_ratio)
    
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    for img in train_images:
        shutil.copy(
            os.path.join(source_dir, img),
            os.path.join(train_dir, img)
        )
        
    for img in val_images:
        shutil.copy(
            os.path.join(source_dir, img),
            os.path.join(val_dir, img)
        )