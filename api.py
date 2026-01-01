"""
FastAPI Server for ML Trust Score System
Deploy this to Render.com
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import os
from datetime import datetime
import mlflow
import dagshub
import pandas as pd
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
    input_data: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    model_id: str
    total_predictions: int
    avg_trust_score: float
    min_trust_score: float
    max_trust_score: float
    trust_breakdown: Dict[str, float]
    explanation: str
    timestamp: str
    sample_predictions: Optional[List[Any]] = None

class UploadResponse(BaseModel):
    model_id: str
    message: str
    supported_format: str
    data_samples: int
    features: List[str]


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "ML Trust Score System API",
        "status": "running",
        "dagshub_connected": bool(DAGSHUB_TOKEN),
        "version": "1.0.0",
        "endpoints": {
            "POST /upload": "Upload model + data",
            "POST /predict": "Make predictions on all data with trust scoring",
            "GET /models": "List all uploaded models",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health():
    """Health check for Render"""
    return {"status": "healthy"}


@app.post("/upload", response_model=UploadResponse)
async def upload_model(
    model_file: UploadFile = File(...),
    data_file: UploadFile = File(...)
):
    """
    Upload model + data with automatic metadata extraction
    
    Args:
        model_file: Model file (.pkl, .pt, .h5)
        data_file: Training data CSV
    
    Returns:
        Upload confirmation with model_id
    """
    
    with mlflow.start_run(run_name=f"upload_{datetime.now().strftime('%H%M%S')}"):
        try:
            # Validate model
            model_format = validate_model_format(model_file.filename)
            if not model_format:
                raise HTTPException(status_code=400, detail="Unsupported model format. Use .pkl, .pt, or .h5")
            
            # Validate and get data info
            data_samples = await validate_data_format(data_file)
            
            # Extract features from CSV
            content = await data_file.read()
            await data_file.seek(0)
            df = pd.read_csv(pd.io.common.BytesIO(content))
            feature_order = df.columns.tolist()
            
            model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save files
            model_path = await save_uploaded_model(model_file, model_id, MODEL_DIR)
            data_path = await save_uploaded_data(data_file, model_id, DATA_DIR)
            
            # Create metadata
            metadata = {
                "model_id": model_id,
                "model_format": model_format,
                "upload_timestamp": datetime.now().isoformat(),
                "model_path": model_path,
                "data_path": data_path,
                "data_samples": data_samples,
                "feature_order": feature_order
            }
            
            metadata_path = os.path.join(METADATA_DIR, f"{model_id}_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
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
                data_samples=data_samples,
                features=feature_order
            )
            
        except Exception as e:
            mlflow.log_metric("upload_success", 0)
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make predictions on all data rows + calculate aggregate trust score
    
    Args:
        request: Model ID (input_data is optional and ignored)
    
    Returns:
        Aggregate trust score across all predictions
    """
    
    with mlflow.start_run(run_name=f"predict_{request.model_id[:8]}"):
        try:
            # Load model
            model, metadata = load_model_from_storage(request.model_id, MODEL_DIR, METADATA_DIR)
            
            # Load data
            data_path = metadata.get("data_path")
            df = pd.read_csv(data_path)
            feature_order = metadata.get("feature_order")
            total_rows = len(df)
            
            # Make predictions on all rows
            all_trust_scores = []
            all_predictions = []
            
            for idx in range(total_rows):
                input_data = {feature: float(df[feature].iloc[idx]) for feature in feature_order}
                
                # Make prediction
                prediction, confidence = make_prediction(model, input_data, metadata)
                all_predictions.append(prediction)
                
                # Calculate trust score
                trust_result = calculate_trust_score(
                    model=model,
                    input_data=input_data,
                    metadata=metadata,
                    model_id=request.model_id,
                    data_dir=DATA_DIR
                )
                all_trust_scores.append(trust_result["trust_score"])
            
            # Calculate aggregate metrics
            avg_trust_score = sum(all_trust_scores) / len(all_trust_scores)
            min_trust_score = min(all_trust_scores)
            max_trust_score = max(all_trust_scores)
            
            # Get sample breakdown
            sample_input = {feature: float(df[feature].iloc[0]) for feature in feature_order}
            trust_result = calculate_trust_score(
                model=model,
                input_data=sample_input,
                metadata=metadata,
                model_id=request.model_id,
                data_dir=DATA_DIR
            )
            
            # Generate explanation
            if avg_trust_score >= 80:
                explanation = "HIGH TRUST (80-100): Model predictions are reliable"
            elif avg_trust_score >= 50:
                explanation = "MEDIUM TRUST (50-80): Use predictions with caution"
            else:
                explanation = "LOW TRUST (0-50): Do NOT rely on these predictions"
            
            # MLflow logging
            mlflow.log_param("model_id", request.model_id)
            mlflow.log_metric("avg_trust_score", avg_trust_score)
            mlflow.log_metric("min_trust_score", min_trust_score)
            mlflow.log_metric("max_trust_score", max_trust_score)
            mlflow.log_metric("total_predictions", total_rows)
            for metric, value in trust_result["breakdown"].items():
                mlflow.log_metric(f"trust_{metric}", value)
            mlflow.log_metric("prediction_success", 1)
            
            # DVC tracking
            log_to_dvc("predict", {
                "model_id": request.model_id,
                "avg_trust_score": avg_trust_score,
                "total_predictions": total_rows
            })
            
            return PredictionResponse(
                model_id=request.model_id,
                total_predictions=total_rows,
                avg_trust_score=round(avg_trust_score, 2),
                min_trust_score=round(min_trust_score, 2),
                max_trust_score=round(max_trust_score, 2),
                trust_breakdown={k: round(v, 2) for k, v in trust_result["breakdown"].items()},
                explanation=explanation,
                timestamp=datetime.now().isoformat(),
                sample_predictions=all_predictions[:5]  # First 5 predictions as sample
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
                        "data_samples": metadata.get("data_samples"),
                        "features": metadata.get("feature_order", [])
                    })
        return {"models": models, "count": len(models)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# For running with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)