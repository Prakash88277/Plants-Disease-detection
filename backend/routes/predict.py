import time
import hashlib
import numpy as np
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from model.model_loader import model_loader
from utils.preprocess import process_image
from config.settings import CLASS_LABELS

router = APIRouter()
logger = logging.getLogger(__name__)

def parse_label(label: str):
    normalized = label.replace("___", "_").replace("__", "_")
    parts = normalized.split("_", 1)
    
    if len(parts) == 2:
        plant_type = parts[0]
        disease = parts[1].replace("_", " ")
        if disease.lower() == "healthy":
            disease = "Healthy"
    else:
        plant_type = label
        disease = "Unknown"
        
    return plant_type, disease

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    try:
        contents = await file.read()
        
        # Diagnostic 1: Confirm backend receives new image
        img_hash = hashlib.md5(contents).hexdigest()
        print(f"--- PREDICTION REQUEST START ---")
        print(f"Image hash: {img_hash}")
        
        start_time = time.time()
        
        img_array = process_image(contents)
        
        model = model_loader.get_model()
        predictions = model.predict(img_array)
        
        print(f"Raw Predictions: {predictions.tolist()}")
        
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        
        print(f"Predicted Index: {predicted_class}")
        print(f"Predicted Label: {CLASS_LABELS[predicted_class]}")
        print(f"Confidence: {confidence}%")
        print(f"--- PREDICTION REQUEST END ---")
        
        label = CLASS_LABELS[predicted_class]
        plant, disease = parse_label(label)
        
        confidence_str = f"{confidence:.2f}%"
        prediction_time_ms = round((time.time() - start_time) * 1000, 2)
        
        result = {
            "plant": plant,
            "disease": disease,
            "confidence": confidence_str,
            "prediction_time_ms": prediction_time_ms,
            "model_name": "Deep Learning CNN"
        }
        
        # Disable caching dynamically
        return JSONResponse(
            content=result, 
            headers={"Cache-Control": "no-store"}
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
