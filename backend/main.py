import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from routes.predict import router as predict_router
from schemas.response_schema import HealthCheckResponse
from model.model_loader import model_loader

# Setup Basic Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="Plant Disease Detection System",
    description="API for classifying plant diseases using a Convolutional Neural Network.",
    version="1.0.0"
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model asynchronously (or concurrently on startup)
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Plant Disease Detection API...")
    try:
        model_loader.load_model()
    except Exception as e:
        logger.error(f"Failed to initialize model on startup: {str(e)}")

# Add Routes
app.include_router(predict_router, tags=["Prediction"])

# Base Route for Health Check
@app.get("/", response_model=HealthCheckResponse, tags=["Health"])
def root():
    return HealthCheckResponse()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
