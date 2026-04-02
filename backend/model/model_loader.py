import os
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class ModelLoader:
    _instance = None
    _resnet_model = None
    _efficientnet_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def load_models(self):
        """Loads both Keras models synchronously using a Singleton pattern."""
        if self._resnet_model is not None and self._efficientnet_model is not None:
            return
            
        import tensorflow.keras as keras
        from tensorflow.keras.models import load_model
        
        try:
            # We explicitly add compile=False to avoid custom object loading issues and speed up inference model loading
            logger.info("Loading EfficientNetB0...")
            self._efficientnet_model = load_model("saved_models/efficientnet_model.h5", compile=False)
            logger.info("✅ EfficientNetB0 Loaded Successfully")
            
            logger.info("Loading ResNet50...")
            self._resnet_model = keras.models.load_model("saved_models/resnet_model.h5", compile=False)
            logger.info("✅ ResNet50 Loaded Successfully")
        except Exception as e:
            try:
                # Fallback paths
                logger.info("Trying fallback absolute paths...")
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                eff_path = os.path.join(base_dir, "saved_models", "efficientnet_model.h5")
                res_path = os.path.join(base_dir, "saved_models", "resnet_model.h5")
                self._efficientnet_model = load_model(eff_path, compile=False)
                self._resnet_model = keras.models.load_model(res_path, compile=False)
                logger.info("✅ Models Loaded Successfully from fallback paths")
            except Exception as e2:
                logger.error(f"Error loading models: {str(e2)}")
                raise e2

    def get_resnet_model(self):
        """Returns the loaded ResNet50 model."""
        if self._resnet_model is None:
            raise HTTPException(status_code=503, detail="ResNet50 Model is currently unavailable or failed to load.")
        return self._resnet_model

    def get_efficientnet_model(self):
        """Returns the loaded EfficientNetB0 model."""
        if self._efficientnet_model is None:
            raise HTTPException(status_code=503, detail="EfficientNetB0 Model is currently unavailable or failed to load.")
        return self._efficientnet_model

# Instantiate the singleton class
model_loader = ModelLoader()
