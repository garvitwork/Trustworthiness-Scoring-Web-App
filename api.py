"""
FastAPI Backend for ML Trust Score System
Handles model upload, prediction, and trust score calculation
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import os
from datetime import datetime

from functions import (
    save_uploaded_model,
    save_uploaded_data,
    validate_model_format,
    validate_data_format,
    load_model_from_storage,
    make_prediction,
    calculate_trust_score
)

app = FastAPI(title="ML Trust Score API", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class PredictionRequest(BaseModel):
    model_id: str
    input_data: Dict[str, Any]

class PredictionResponse(BaseModel):
    model_id: str
    prediction: Any
    confidence: Optional[float]
    trust_score: float
    trust_breakdown: Dict[str, float]
    explanation: str
    timestamp: str

class UploadResponse(BaseModel):
    model_id: str
    message: str
    supported_format: str
    data_samples: int

# Storage paths
UPLOAD_DIR = "uploads"
MODEL_DIR = os.path.join(UPLOAD_DIR, "models")
DATA_DIR = os.path.join(UPLOAD_DIR, "data")
METADATA_DIR = os.path.join(UPLOAD_DIR, "metadata")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "ML Trust Score System API",
        "status": "running",
        "version": "1.0.0"
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_model(
    model_file: UploadFile = File(...),
    data_file: UploadFile = File(...),
    metadata: UploadFile = File(...)
):
    """
    Upload model, training data, and metadata
    
    Expected files:
    - model_file: .pkl (scikit-learn), .pt (PyTorch), or SavedModel folder (TensorFlow)
    - data_file: .csv with training data
    - metadata: .json with training parameters
    """
    try:
        # Read metadata
        metadata_content = await metadata.read()
        metadata_json = json.loads(metadata_content)
        
        # Generate unique model ID
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Validate model format
        model_format = validate_model_format(model_file.filename)
        if not model_format:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model format. Supported: .pkl, .pt, SavedModel"
            )
        
        # Save model
        model_path = await save_uploaded_model(model_file, model_id, MODEL_DIR)
        
        # Validate and save data
        data_samples = await validate_data_format(data_file)
        data_path = await save_uploaded_data(data_file, model_id, DATA_DIR)
        
        # Save metadata
        metadata_json["model_id"] = model_id
        metadata_json["model_format"] = model_format
        metadata_json["upload_timestamp"] = datetime.now().isoformat()
        metadata_json["model_path"] = model_path
        metadata_json["data_path"] = data_path
        
        metadata_path = os.path.join(METADATA_DIR, f"{model_id}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata_json, f, indent=2)
        
        return UploadResponse(
            model_id=model_id,
            message="Model uploaded successfully",
            supported_format=model_format,
            data_samples=data_samples
        )
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make prediction and calculate trust score
    
    Input:
    - model_id: ID of uploaded model
    - input_data: Dictionary with input features
    
    Output:
    - prediction: Model output
    - confidence: Model confidence (if available)
    - trust_score: Combined trust score (0-100)
    - trust_breakdown: Individual metric scores
    - explanation: Human-readable trust explanation
    """
    try:
        # Load model and metadata
        model, metadata = load_model_from_storage(request.model_id, MODEL_DIR, METADATA_DIR)
        
        # Make prediction
        prediction, confidence = make_prediction(model, request.input_data, metadata)
        
        # Calculate trust score
        trust_result = calculate_trust_score(
            model=model,
            input_data=request.input_data,
            metadata=metadata,
            model_id=request.model_id,
            data_dir=DATA_DIR
        )
        
        return PredictionResponse(
            model_id=request.model_id,
            prediction=prediction,
            confidence=confidence,
            trust_score=trust_result["trust_score"],
            trust_breakdown=trust_result["breakdown"],
            explanation=trust_result["explanation"],
            timestamp=datetime.now().isoformat()
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/trust-score/{model_id}")
async def get_trust_metrics(model_id: str):
    """
    Get trust metrics breakdown for a specific model
    """
    try:
        metadata_path = os.path.join(METADATA_DIR, f"{model_id}_metadata.json")
        
        if not os.path.exists(metadata_path):
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        return {
            "model_id": model_id,
            "model_format": metadata.get("model_format"),
            "upload_timestamp": metadata.get("upload_timestamp"),
            "training_params": metadata.get("training_params", {}),
            "available_metrics": [
                "confidence_consistency",
                "data_familiarity",
                "agreement_score",
                "explanation_stability",
                "historical_error_rate"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List all uploaded models"""
    try:
        models = []
        for filename in os.listdir(METADATA_DIR):
            if filename.endswith("_metadata.json"):
                with open(os.path.join(METADATA_DIR, filename), "r") as f:
                    metadata = json.load(f)
                    models.append({
                        "model_id": metadata.get("model_id"),
                        "model_format": metadata.get("model_format"),
                        "upload_timestamp": metadata.get("upload_timestamp")
                    })
        return {"models": models, "count": len(models)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)