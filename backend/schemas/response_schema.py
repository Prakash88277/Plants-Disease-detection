from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    plant: str = Field(..., description="The type of plant predicted.")
    disease: str = Field(..., description="The name of the disease or 'healthy'.")
    confidence: str = Field(..., description="Confidence percentage formatted as a string (e.g. '96.7%').")
    prediction_time_ms: float = Field(..., description="Time taken to make the prediction in milliseconds.")
    model_name: str = Field(default="CNN Model", description="Name of the model used.")

class HealthCheckResponse(BaseModel):
    status: str = Field(default="Running", description="Health status of the API")
    message: str = Field(default="Plant Disease Detection API Running", description="A welcome message")
