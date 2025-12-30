
# load used libraries
from .data.dataset_builder import split_dataset
import configs.config as config

def main():
    classes = ['hijab', 'without_hijab']
    
    for cls in classes:
        split_dataset(
            source_dir = f'{config.ORIGINAL_DATASET_DIR}/{cls}',
            train_dir = f'{config.DATASET_DIR}/train/{cls}',
            val_dir = f'{config.DATASET_DIR}/val/{cls}'
        )
        
        
if __name__ == '__main__':
    main()