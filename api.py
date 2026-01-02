"""
FastAPI Server for ML Trust Score System
Deploy this to Render.com - WITH BATCH PROCESSING AND FIXED DAGSHUB TRACKING
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import traceback
import uuid
import asyncio

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

# Set DagHub authentication BEFORE any DagHub imports
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "f43d18eaef53b8269dd37f6434a8612b4faa6c8b")
DAGSHUB_USER = os.getenv("DAGSHUB_USER", "garvitwork")
if DAGSHUB_TOKEN:
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

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
JOBS_DIR = "jobs"

# Create directories
for d in [MODEL_DIR, DATA_DIR, METADATA_DIR, JOBS_DIR]:
    os.makedirs(d, exist_ok=True)

# Job storage for async processing
prediction_jobs = {}

# Get DagHub config
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO", "garvitwork/ml-trust-score")
MLFLOW_ENABLED = False

def initialize_mlflow():
    """Initialize MLflow with proper DagHub authentication"""
    global MLFLOW_ENABLED
    
    try:
        import mlflow
        import dagshub
        
        if DAGSHUB_TOKEN:
            try:
                # Initialize DagHub
                dagshub.init(
                    repo_owner=DAGSHUB_REPO.split('/')[0],
                    repo_name=DAGSHUB_REPO.split('/')[1],
                    mlflow=True
                )
                
                # Set tracking URI with authentication
                tracking_uri = f"https://dagshub.com/{DAGSHUB_REPO}.mlflow"
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment("ml-trust-score")
                
                # Test connection with a dummy run
                with mlflow.start_run(run_name="connection_test") as run:
                    mlflow.log_param("test", "connection")
                    mlflow.log_metric("status", 1)
                
                MLFLOW_ENABLED = True
                print(f"✅ DagHub connected: {DAGSHUB_REPO}")
                print(f"🔗 Tracking URI: {tracking_uri}")
                return True
                
            except Exception as e:
                print(f"⚠️ DagHub connection failed: {e}")
                print(f"📋 Traceback: {traceback.format_exc()}")
                
                # Fallback to local MLflow
                mlflow.set_tracking_uri("file://./mlruns")
                MLFLOW_ENABLED = True
                print("⚠️ Using local MLflow as fallback")
                return False
        else:
            print("⚠️ DAGSHUB_TOKEN not found in environment")
            mlflow.set_tracking_uri("file://./mlruns")
            MLFLOW_ENABLED = True
            print("⚠️ Using local MLflow")
            return False
            
    except Exception as e:
        print(f"⚠️ MLflow initialization failed: {e}")
        MLFLOW_ENABLED = False
        return False

# Initialize on startup
mlflow_connected = initialize_mlflow()

# Pydantic Models
class PredictionRequest(BaseModel):
    model_id: str
    batch_size: Optional[int] = 100

class PredictionResponse(BaseModel):
    job_id: str
    model_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    total_predictions: Optional[int] = None
    processed_predictions: Optional[int] = None
    avg_trust_score: Optional[float] = None
    min_trust_score: Optional[float] = None
    max_trust_score: Optional[float] = None
    trust_breakdown: Optional[Dict[str, float]] = None
    explanation: Optional[str] = None
    timestamp: Optional[str] = None
    error: Optional[str] = None
    mlflow_logged: Optional[bool] = None  # Track if MLflow logging succeeded

class UploadResponse(BaseModel):
    model_id: str
    message: str
    supported_format: str
    data_samples: int
    features: List[str]
    mlflow_logged: Optional[bool] = None


def log_to_mlflow(operation: str, data: Dict[str, Any], model_id: str = None):
    """
    Centralized MLflow logging function with robust error handling
    
    Args:
        operation: "upload" or "predict"
        data: Dictionary of metrics/parameters to log
        model_id: Optional model ID for run naming
    
    Returns:
        bool: True if logging succeeded, False otherwise
    """
    if not MLFLOW_ENABLED:
        print("⚠️ MLflow not enabled, skipping logging")
        return False
    
    try:
        import mlflow
        
        # Create run name
        run_name = f"{operation}_{model_id[:8] if model_id else datetime.now().strftime('%H%M%S')}"
        
        # Start MLflow run
        with mlflow.start_run(run_name=run_name) as run:
            # Log operation type
            mlflow.log_param("operation", operation)
            mlflow.log_param("timestamp", datetime.now().isoformat())
            
            if operation == "upload":
                # Log upload-specific data
                mlflow.log_param("model_id", data.get("model_id"))
                mlflow.log_param("model_format", data.get("model_format"))
                mlflow.log_param("data_samples", data.get("data_samples"))
                mlflow.log_metric("upload_success", 1)
                
                if data.get("features"):
                    mlflow.log_param("num_features", len(data["features"]))
                    mlflow.log_param("features", ",".join(data["features"][:5]))  # First 5 features
            
            elif operation == "predict":
                # Log prediction-specific data
                mlflow.log_param("model_id", data.get("model_id"))
                mlflow.log_metric("avg_trust_score", data.get("avg_trust_score", 0))
                mlflow.log_metric("min_trust_score", data.get("min_trust_score", 0))
                mlflow.log_metric("max_trust_score", data.get("max_trust_score", 0))
                mlflow.log_metric("total_predictions", data.get("total_predictions", 0))
                
                # Log breakdown metrics
                if data.get("trust_breakdown"):
                    for metric, value in data["trust_breakdown"].items():
                        mlflow.log_metric(f"trust_{metric}", value)
                
                mlflow.log_metric("prediction_success", 1)
            
            print(f"✅ MLflow logged: {operation} - Run ID: {run.info.run_id}")
            return True
            
    except Exception as e:
        print(f"⚠️ MLflow logging failed for {operation}: {e}")
        print(f"📋 Traceback: {traceback.format_exc()}")
        return False


def process_predictions_batch(job_id: str, model_id: str, batch_size: int):
    """Background task to process predictions in batches"""
    
    job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
    mlflow_logged = False
    
    try:
        # Update status to processing
        job_data = {
            "job_id": job_id,
            "status": "processing",
            "progress": 0,
            "model_id": model_id,
            "started_at": datetime.now().isoformat()
        }
        with open(job_file, "w") as f:
            json.dump(job_data, f, indent=2)
        
        # Load model
        print(f"📦 [Job {job_id}] Loading model...")
        model, metadata = load_model_from_storage(model_id, MODEL_DIR, METADATA_DIR)
        
        # Load data
        data_path = metadata.get("data_path")
        df = pd.read_csv(data_path)
        feature_order = metadata.get("feature_order")
        total_rows = len(df)
        
        print(f"📊 [Job {job_id}] Processing {total_rows} predictions in batches of {batch_size}...")
        
        all_trust_scores = []
        all_predictions = []
        
        # Process in batches
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            
            # Process batch
            for idx in range(start_idx, end_idx):
                input_data = {feature: float(df[feature].iloc[idx]) for feature in feature_order}
                
                # Make prediction
                prediction, confidence = make_prediction(model, input_data, metadata)
                all_predictions.append(prediction)
                
                # Calculate trust score
                trust_result = calculate_trust_score(
                    model=model,
                    input_data=input_data,
                    metadata=metadata,
                    model_id=model_id,
                    data_dir=DATA_DIR
                )
                all_trust_scores.append(trust_result["trust_score"])
            
            # Update progress
            progress = (end_idx / total_rows) * 100
            job_data["progress"] = round(progress, 2)
            job_data["processed_predictions"] = end_idx
            job_data["total_predictions"] = total_rows
            
            with open(job_file, "w") as f:
                json.dump(job_data, f, indent=2)
            
            print(f"  Progress: {end_idx}/{total_rows} ({progress:.1f}%)")
        
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
            model_id=model_id,
            data_dir=DATA_DIR
        )
        
        # Generate explanation
        if avg_trust_score >= 80:
            explanation = "HIGH TRUST (80-100): Model predictions are reliable"
        elif avg_trust_score >= 50:
            explanation = "MEDIUM TRUST (50-80): Use predictions with caution"
        else:
            explanation = "LOW TRUST (0-50): Do NOT rely on these predictions"
        
        # MLflow logging (with retry)
        print(f"📤 [Job {job_id}] Logging to MLflow/DagHub...")
        mlflow_logged = log_to_mlflow(
            operation="predict",
            data={
                "model_id": model_id,
                "avg_trust_score": avg_trust_score,
                "min_trust_score": min_trust_score,
                "max_trust_score": max_trust_score,
                "total_predictions": total_rows,
                "trust_breakdown": trust_result["breakdown"]
            },
            model_id=model_id
        )
        
        if mlflow_logged:
            print(f"✅ [Job {job_id}] MLflow logging successful")
        else:
            print(f"⚠️ [Job {job_id}] MLflow logging failed (results still saved)")
        
        # DVC tracking (non-blocking)
        try:
            log_to_dvc("predict", {
                "model_id": model_id,
                "avg_trust_score": avg_trust_score,
                "total_predictions": total_rows
            })
        except Exception as e:
            print(f"⚠️ DVC tracking failed: {e}")
        
        # Update job as completed
        job_data["status"] = "completed"
        job_data["progress"] = 100
        job_data["total_predictions"] = total_rows
        job_data["processed_predictions"] = total_rows
        job_data["avg_trust_score"] = round(avg_trust_score, 2)
        job_data["min_trust_score"] = round(min_trust_score, 2)
        job_data["max_trust_score"] = round(max_trust_score, 2)
        job_data["trust_breakdown"] = {k: round(v, 2) for k, v in trust_result["breakdown"].items()}
        job_data["explanation"] = explanation
        job_data["timestamp"] = datetime.now().isoformat()
        job_data["completed_at"] = datetime.now().isoformat()
        job_data["mlflow_logged"] = mlflow_logged
        
        with open(job_file, "w") as f:
            json.dump(job_data, f, indent=2)
        
        print(f"✅ [Job {job_id}] Completed successfully")
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(f"❌ [Job {job_id}] {error_msg}")
        print(traceback.format_exc())
        
        # Update job as failed
        job_data = {
            "job_id": job_id,
            "status": "failed",
            "error": error_msg,
            "failed_at": datetime.now().isoformat(),
            "mlflow_logged": False
        }
        with open(job_file, "w") as f:
            json.dump(job_data, f, indent=2)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "ML Trust Score System API",
        "status": "running",
        "dagshub_repo": DAGSHUB_REPO,
        "dagshub_connected": mlflow_connected,
        "mlflow_enabled": MLFLOW_ENABLED,
        "version": "2.1.0",
        "features": ["batch_processing", "async_predictions", "progress_tracking", "dagshub_tracking"],
        "endpoints": {
            "POST /upload": "Upload model + data",
            "POST /predict": "Start prediction job (async)",
            "GET /job/{job_id}": "Check prediction job status",
            "GET /models": "List all uploaded models",
            "GET /health": "Health check",
            "GET /mlflow-status": "Check MLflow connection"
        }
    }


@app.get("/health")
async def health():
    """Health check for Render"""
    return {
        "status": "healthy",
        "mlflow_enabled": MLFLOW_ENABLED,
        "dagshub_connected": mlflow_connected,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/mlflow-status")
async def mlflow_status():
    """Check MLflow/DagHub connection status"""
    return {
        "mlflow_enabled": MLFLOW_ENABLED,
        "dagshub_connected": mlflow_connected,
        "dagshub_repo": DAGSHUB_REPO,
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "Not set"),
        "has_token": bool(DAGSHUB_TOKEN),
        "has_user": bool(DAGSHUB_USER)
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_model(
    model_file: UploadFile = File(...),
    data_file: UploadFile = File(...)
):
    """
    Upload model + data with automatic metadata extraction
    """
    
    print(f"📁 Upload request received - Model: {model_file.filename}, Data: {data_file.filename}")
    
    mlflow_logged = False
    
    try:
        # Validate model
        print("🔍 Validating model format...")
        model_format = validate_model_format(model_file.filename)
        if not model_format:
            raise HTTPException(status_code=400, detail="Unsupported model format. Use .pkl, .pt, or .h5")
        print(f"✓ Model format: {model_format}")
        
        # Validate and get data info
        print("🔍 Validating data format...")
        data_samples = await validate_data_format(data_file)
        print(f"✓ Data samples: {data_samples}")
        
        # Extract features from CSV
        print("🔍 Extracting features from CSV...")
        content = await data_file.read()
        await data_file.seek(0)
        df = pd.read_csv(pd.io.common.BytesIO(content))
        feature_order = df.columns.tolist()
        print(f"✓ Features: {', '.join(feature_order)}")
        
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"🆔 Generated model_id: {model_id}")
        
        # Save files
        print("💾 Saving model file...")
        model_path = await save_uploaded_model(model_file, model_id, MODEL_DIR)
        print(f"✓ Model saved: {model_path}")
        
        print("💾 Saving data file...")
        data_path = await save_uploaded_data(data_file, model_id, DATA_DIR)
        print(f"✓ Data saved: {data_path}")
        
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
        print(f"✓ Metadata saved: {metadata_path}")
        
        # MLflow logging
        print(f"📤 Logging upload to MLflow/DagHub...")
        mlflow_logged = log_to_mlflow(
            operation="upload",
            data={
                "model_id": model_id,
                "model_format": model_format,
                "data_samples": data_samples,
                "features": feature_order
            },
            model_id=model_id
        )
        
        if mlflow_logged:
            print("✅ MLflow logging completed")
        else:
            print("⚠️ MLflow logging failed (upload still successful)")
        
        # DVC tracking (non-blocking)
        try:
            log_to_dvc("upload", {"model_id": model_id, "samples": data_samples})
            print("✓ DVC tracking completed")
        except Exception as e:
            print(f"⚠️ DVC tracking failed (non-critical): {e}")
        
        print(f"✅ Upload completed successfully: {model_id}")
        
        return UploadResponse(
            model_id=model_id,
            message="Model uploaded successfully",
            supported_format=model_format,
            data_samples=data_samples,
            features=feature_order,
            mlflow_logged=mlflow_logged
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Upload failed: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Start async prediction job for all data rows
    """
    
    print(f"📊 Prediction request for model: {request.model_id}")
    
    try:
        # Verify model exists
        metadata_path = os.path.join(METADATA_DIR, f"{request.model_id}_metadata.json")
        if not os.path.exists(metadata_path):
            raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
        
        # Create job
        job_id = str(uuid.uuid4())
        print(f"🆔 Created job: {job_id}")
        
        # Initialize job status
        job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
        job_data = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0,
            "model_id": request.model_id,
            "created_at": datetime.now().isoformat()
        }
        with open(job_file, "w") as f:
            json.dump(job_data, f, indent=2)
        
        # Start background task
        batch_size = request.batch_size or 100
        background_tasks.add_task(process_predictions_batch, job_id, request.model_id, batch_size)
        
        print(f"✅ Job started: {job_id}")
        
        return PredictionResponse(
            job_id=job_id,
            model_id=request.model_id,
            status="pending",
            message=f"Prediction job started. Use /job/{job_id} to check progress."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to start prediction job: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of a prediction job"""
    
    job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
    
    if not os.path.exists(job_file):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    try:
        with open(job_file, "r") as f:
            job_data = json.load(f)
        
        return JobStatusResponse(**job_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read job status: {str(e)}")


@app.get("/models")
async def list_models():
    """List all uploaded models"""
    try:
        print("📋 Listing models...")
        models = []
        
        if not os.path.exists(METADATA_DIR):
            return {"models": models, "count": 0}
        
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
        
        print(f"✓ Found {len(models)} models")
        return {"models": models, "count": len(models)}
    except Exception as e:
        print(f"❌ List models failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# For running with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)