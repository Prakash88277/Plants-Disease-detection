import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import HTTPException

def process_image(file_bytes: bytes) -> np.ndarray:
    try:
        # Match EXACT training preprocessing
        img = Image.open(BytesIO(file_bytes)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format or failed to process image: {str(e)}")
