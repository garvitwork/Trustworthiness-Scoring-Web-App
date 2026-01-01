"""
FastAPI Server for ML Trust Score System
This runs independently from main.py and will be deployed to Render
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import os
from datetime import datetime
import mlflow
import dagshub
from dotenv import load_dotenv

from functions import (
    save_uploaded_model,
    save_uploaded_data,
    validate_model_format,
    validate_data_format,
    load_model_from_storage,
    make_prediction,
    calculate_trust_score,
    log_to_dvc
)

load_dotenv()

app = FastAPI(title="ML Trust Score API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage paths
MODEL_DIR = "uploads/models"
DATA_DIR = "uploads/data"
METADATA_DIR = "uploads/metadata"

# Create directories
for d in [MODEL_DIR, DATA_DIR, METADATA_DIR]:
    os.makedirs(d, exist_ok=True)

# Initialize DagHub + MLflow
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO", "garvitwork/ml-trust-score")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "")

if DAGSHUB_TOKEN:
    try:
        dagshub.init(
            repo_owner=DAGSHUB_REPO.split('/')[0],
            repo_name=DAGSHUB_REPO.split('/')[1],
            mlflow=True
        )
        mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_REPO}.mlflow")
        mlflow.set_experiment("ml-trust-score")
        print(f"✓ DagHub connected: {DAGSHUB_REPO}")
    except:
        mlflow.set_tracking_uri("file://./mlruns")
        print("⚠ Using local MLflow")
else:
    mlflow.set_tracking_uri("file://./mlruns")
    print("⚠ Using local MLflow")

# Pydantic Models
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


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "ML Trust Score System API",
        "status": "running",
        "dagshub_connected": bool(DAGSHUB_TOKEN),
        "version": "1.0.0"
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_model(
    model_file: UploadFile = File(...),
    data_file: UploadFile = File(...),
    metadata: UploadFile = File(...)
):
    """
    Upload model + data + metadata with DVC/MLflow tracking
    
    Args:
        model_file: Model file (.pkl, .pt, .h5)
        data_file: Training data CSV
        metadata: Metadata JSON with feature_order
    
    Returns:
        Upload confirmation with model_id
    """
    
    with mlflow.start_run(run_name=f"upload_{datetime.now().strftime('%H%M%S')}"):
        try:
            # Parse metadata
            metadata_content = await metadata.read()
            metadata_json = json.loads(metadata_content)
            
            model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Validate model
            model_format = validate_model_format(model_file.filename)
            if not model_format:
                raise HTTPException(status_code=400, detail="Unsupported model format")
            
            # Save files
            model_path = await save_uploaded_model(model_file, model_id, MODEL_DIR)
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
            
            # MLflow logging
            mlflow.log_param("model_id", model_id)
            mlflow.log_param("model_format", model_format)
            mlflow.log_param("data_samples", data_samples)
            mlflow.log_artifact(metadata_path)
            mlflow.log_metric("upload_success", 1)
            
            # DVC tracking
            log_to_dvc("upload", {"model_id": model_id, "samples": data_samples})
            
            return UploadResponse(
                model_id=model_id,
                message="Model uploaded successfully",
                supported_format=model_format,
                data_samples=data_samples
            )
            
        except Exception as e:
            mlflow.log_metric("upload_success", 0)
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make prediction + calculate trust score with tracking
    
    Args:
        request: Model ID and input data
    
    Returns:
        Prediction with trust score breakdown
    """
    
    with mlflow.start_run(run_name=f"predict_{request.model_id[:8]}"):
        try:
            # Load model
            model, metadata = load_model_from_storage(request.model_id, MODEL_DIR, METADATA_DIR)
            
            # Log inputs
            mlflow.log_param("model_id", request.model_id)
            for k, v in request.input_data.items():
                mlflow.log_param(f"input_{k}", v)
            
            # Predict
            prediction, confidence = make_prediction(model, request.input_data, metadata)
            
            # Calculate trust
            trust_result = calculate_trust_score(
                model=model,
                input_data=request.input_data,
                metadata=metadata,
                model_id=request.model_id,
                data_dir=DATA_DIR
            )
            
            # MLflow logging
            if confidence:
                mlflow.log_metric("model_confidence", confidence)
            mlflow.log_metric("trust_score", trust_result["trust_score"])
            for metric, value in trust_result["breakdown"].items():
                mlflow.log_metric(f"trust_{metric}", value)
            mlflow.log_metric("prediction_success", 1)
            
            # DVC tracking
            log_to_dvc("predict", {
                "model_id": request.model_id,
                "trust_score": trust_result["trust_score"]
            })
            
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
            mlflow.log_metric("prediction_success", 0)
            raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
        except Exception as e:
            mlflow.log_metric("prediction_success", 0)
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
                        "upload_timestamp": metadata.get("upload_timestamp"),
                        "data_samples": metadata.get("data_samples")
                    })
        return {"models": models, "count": len(models)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# For running with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)