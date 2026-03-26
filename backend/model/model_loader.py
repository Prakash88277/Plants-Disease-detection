import os
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class ModelLoader:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def load_model(self):
        """Loads the Keras model using a Singleton pattern."""
        if self._model is not None:
            return self._model
            
        from tensorflow.keras.models import load_model
        
        try:
            self._model = load_model("saved_models/model.keras")
            print("✅ Model Loaded Successfully")
        except Exception as e:
            try:
                # Fallback for explicit path usage
                fallback_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models", "model.keras")
                self._model = load_model(fallback_path)
                print("✅ Model Loaded Successfully")
            except Exception as e2:
                logger.error(f"Error loading model: {str(e2)}")
                raise e2
            
        return self._model

    def get_model(self):
        """Returns the loaded model, raises an error if not loaded."""
        if self._model is None:
            raise HTTPException(status_code=503, detail="Model is currently unavailable or failed to load.")
        return self._model

# Instantiate the singleton class
model_loader = ModelLoader()
