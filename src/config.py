import os

ENV = "prod"     # [dev/test/prod]

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
print("PROJECT_DIR:\t", PROJECT_DIR)

DATA_FOLDER = os.path.join(PROJECT_DIR, 'data')
MODELS_FOLDER = os.path.join(PROJECT_DIR, 'models')
LOGS_FOLDER = os.path.join(PROJECT_DIR, 'logs')

if ENV == "dev":
    print("DATA_FOLDER:\t", DATA_FOLDER)
    print("MODELS_FOLDER:\t", MODELS_FOLDER)
    print("LOGS_FOLDER:\t", LOGS_FOLDER)

MODEL_WEIGHTS_PATH = os.path.join(MODELS_FOLDER, "bmi_model_weights.h5")

RESNET50_DEFAULT_IMG_WIDTH = 224
MARGIN = .1
TRAIN_BATCH_SIZE = 16
VALIDATION_SIZE = 100

ORIGINAL_IMGS_DIR = 'images'
ORIGINAL_IMGS_INFO_FILE = 'data.csv'
AGE_TRAINED_WEIGHTS_FILE = 'age_only_resnet50_weights.061-3.300-4.410.hdf5'
CROPPED_IMGS_DIR = 'normalized_images'
CROPPED_IMGS_INFO_FILE = 'normalized_data.csv'
TOP_LAYER_LOG_DIR = 'logs/top_layer'
ALL_LAYERS_LOG_DIR = 'logs/all_layers'
