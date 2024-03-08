#################################################################
# CONSTANTS
################################################################# 

PORT = 9159 # Indicates the port of the server
MESSAGE_LENGTH = 100 * 1024 * 1024 # 100 MB messages allowed
MAIN_DIR = "MAIN_DIR" # Environment variable name
DEBUG = True

#################################################################
# PATHS
################################################################# 

# Paths parts
MODELS_DIR = "models"
SHARED_MEMORY_DIR = "shared_memory"


# Paths
MODEL_PATH = lambda model_name: f"{MODELS_DIR}/{model_name}"
SHARED_MEMORY_PATH = lambda shared_memory_name: f"{SHARED_MEMORY_DIR}/{shared_memory_name}"

#################################################################
# MODELS
################################################################# 

# Load model info
MODEL_CONFIG = 'config.json'

# Pytorch models
PYTORCH_MODELS = ['model.pt']

COLOR_PALETTE = {
    0: [138,216,195],
    1: [53,98,236],
    2: [28,109,34],
    }

DEFAULT_MIN_SCORE_MAP = {
    'joint': 0.2,
    'missing': 0.3,
    'stacker': 0.6,
}