import sys
import os
import io
import time
import hashlib
import numpy as np
import logging
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from model.model_loader import model_loader
from config.settings import CLASS_LABELS

from utils.pipeline import predict as cascaded_predict

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    try:
        contents = await file.read()
        
        # Diagnostic: Confirm backend receives new image
        img_hash = hashlib.md5(contents).hexdigest()
        print(f"--- PREDICTION REQUEST START ---")
        print(f"Image hash: {img_hash}")
        
        start_time = time.time()
        
        # Load the image into PIL
        try:
            image = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception as preprocess_err:
            raise HTTPException(status_code=422, detail=f"Image parsing failed: {str(preprocess_err)}")

        # Fetch models synchronously (already loaded)
        resnet_model = model_loader.get_resnet_model()
        efficientnet_model = model_loader.get_efficientnet_model()
        
        # Call the single-source cascaded inference pipeline
        pred_result = cascaded_predict(
            image_input=image,
            resnet_model=resnet_model,
            efficientnet_model=efficientnet_model,
            class_names=CLASS_LABELS,
            threshold=0.30
        )
        
        confidence_str = f"{pred_result['confidence'] * 100:.2f}%"
        stage1_confidence_str = f"{pred_result.get('stage1_confidence', 0) * 100:.2f}%"
        prediction_time_ms = round((time.time() - start_time) * 1000, 2)
        
        print(f"Plant: {pred_result['plant']} | Disease: {pred_result['disease']}")
        print(f"Confidence: {confidence_str} | Stage 1 Conf: {stage1_confidence_str}")
        print(f"--- PREDICTION REQUEST END ---")
        
        result = {
            "plant": pred_result["plant"],
            "disease": pred_result["disease"],
            "confidence": confidence_str,
            "stage1_confidence": stage1_confidence_str,
            "prediction_time_ms": prediction_time_ms,
            "model_name": "Cascaded ResNet50 + EfficientNetB0"
        }
        
        # Disable caching dynamically
        return JSONResponse(
            content=result, 
            headers={"Cache-Control": "no-store"}
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
