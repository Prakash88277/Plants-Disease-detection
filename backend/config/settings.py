import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
# The user might have a specific model name, but the requirements mentioned "saved_models/model.keras"
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")

# API Configuration
API_TITLE = "Plant Disease Detection API"
API_VERSION = "1.0.0"

# Preprocessing Config
IMAGE_SIZE = (224, 224)
CHANNELS = 3

# Class Labels (Sorted alphabetically based on typical folder structures)
CLASS_LABELS = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]
