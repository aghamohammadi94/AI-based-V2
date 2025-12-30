
# load used libraries
import os
from pathlib import Path
from dotenv import load_dotenv


# -------------------------
# Load .env file
# -------------------------
env_path = Path('.') / '.env'
## env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


# -------------------------
# Helper converters
# -------------------------
def _int(name, default):
    val = os.getenv(name)
    
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default


def _float(name, default):
    val = os.getenv(name)
    
    try:
        return float(val) if val is not None else default
    except ValueError:
        return default

    
def _bool(name, default=False):
    val = os.getenv(name)
    
    if val is None:
        return default
    return val.lower() in ('true', '1', 'yes', 'y')


def _tuple(name, default=None, cast=int, expected_len=None):
    val = os.getenv(name)
    
    if val is None:
        val = default

    # if already tuple
    if isinstance(val, tuple):
        return tuple(cast(v) for v in val)

    # must be string now
    if not isinstance(val, str):
        val = default

    # parse
    parts = [p.strip() for p in val.split(",")]

    out = []
    for p in parts:
        try:
            out.append(cast(p))
        except (ValueError, TypeError):
            return _tuple_fallback(default, cast)

    # length validation
    if expected_len is None:
        if isinstance(default, tuple):
            expected_len = len(default)
        else:
            expected_len = len(default.split(","))

    if len(out) != expected_len:
        return _tuple_fallback(default, cast)

    return tuple(out)


def _tuple_fallback(default, cast):
    if isinstance(default, tuple):
        return tuple(cast(v) for v in default)
    return tuple(cast(p.strip()) for p in default.split(","))


# -------------------------
# Read variables from .env
# -------------------------
RAW_IMAGES_DIR = os.getenv('RAW_IMAGES_DIR', '../src/data/downloads')
ORIGINAL_DATASET_DIR = os.getenv('ORIGINAL_DATASET_DIR', '../src/data/images')
DATASET_DIR = os.getenv('DATASET_DIR', '../src/data/datasets')
TEST_DIR = os.getenv('TEST_DIR', '../test-images')
VGG16_WEIGHTS_PATH = os.getenv('VGG16_WEIGHTS_PATH', '../vgg16_notop/vgg16_weights_tf_dim_ordering_tf_dim_ordering_tf_kernels_notop.h5')
VGG16_MODEL_PATH = os.getenv('VGG16_MODEL_PATH', './Hijab_Detection_model_with_vgg16.h5')
PLOTS_DIR = os.getenv('PLOTS_DIR', '../artifacts/plots')
TRAIN_DICTIONARY = os.getenv('TRAIN_DICTIONARY', './train_dictionary.txt')

BATH_SIZE = _int('BATH_SIZE', 20)
NUMBER_TRAINING_IMAGES = _int('NUMBER_TRAINING_IMAGES', 2000)
NUMBER_VALIDATION_IMAGES = _int('NUMBER_VALIDATION_IMAGES', 600)
TARGET_SIZE = _tuple('TARGET_SIZE', default='150,150', cast=int)
LEARNING_RATE = _float('LEARNING_RATE', 0.00002)
STEP_PER_EPOCH = _int('STEP_PER_EPOCH', 100)
EPOCHS = _int('EPOCHS', 40)
VALIDATION_STEPS = _int('VALIDATION_STEPS', 30)
INPUT_SHAPE = _tuple('INPUT_SHAPE', default='150,150,3', cast=int)
MARGIN = _int('MARGIN', 50)

DEBUG = _bool('DEBUG', True)


# -------------------------
# Print config (optional)
# -------------------------
if DEBUG:
    print("\n[CONFIG LOADED]")
    for key in [
        'RAW_IMAGES_DIR', 'ORIGINAL_DATASET_DIR', 'DATASET_DIR',
        'TEST_DIR', 'VGG16_WEIGHTS_PATH', 'VGG16_MODEL_PATH',
        'PLOTS_DIR', 'TRAIN_DICTIONARY', 'BATH_SIZE',
        'NUMBER_TRAINING_IMAGES', 'NUMBER_VALIDATION_IMAGES',
        'TARGET_SIZE', 'LEARNING_RATE', 'STEP_PER_EPOCH', 'EPOCHS',
        'VALIDATION_STEPS', 'INPUT_SHAPE', 'MARGIN'
    ]:
        print(f"{key}: {globals().get(key)}")
