"""
Core Utility Functions for ML Trust Score System
All reusable functions for model handling, trust calculation, and tracking
"""

import os
import json
import pickle
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# DVC & DAGSHUB FUNCTIONS
# ============================================================================

def initialize_dvc(project_dir: Path) -> bool:
    """Initialize DVC in project directory"""
    try:
        os.chdir(project_dir)
        
        # Check if already initialized
        if (project_dir / ".dvc").exists():
            logger.info("DVC already initialized")
            return True
        
        # Initialize DVC
        result = subprocess.run(["dvc", "init"], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("DVC initialized successfully")
            
            # Configure DVC
            subprocess.run(["dvc", "config", "core.autostage", "true"])
            
            return True
        else:
            logger.error(f"DVC init failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        logger.error("DVC not installed. Install with: pip install dvc")
        return False
    except Exception as e:
        logger.error(f"DVC initialization error: {e}")
        return False


def setup_dagshub_connection(repo_url: str, token: str = None) -> bool:
    """Setup DagHub remote for DVC"""
    try:
        # Add DagHub as DVC remote
        remote_url = repo_url.replace("https://", f"https://{token}@") if token else repo_url
        
        result = subprocess.run(
            ["dvc", "remote", "add", "-d", "dagshub", remote_url],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 or "already exists" in result.stderr:
            logger.info("DagHub remote configured")
            return True
        else:
            logger.warning(f"DagHub remote setup: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"DagHub connection error: {e}")
        return False


def create_pipeline_structure(base_dir: Path) -> None:
    """Create standard MLOps directory structure"""
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "metrics",
        "logs",
        "notebooks",
        "src",
        "tests"
    ]
    
    for dir_path in directories:
        (base_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created project structure in {base_dir}")


# ============================================================================
# MODEL UPLOAD & VALIDATION FUNCTIONS
# ============================================================================

def validate_model_format(filename: str) -> Optional[str]:
    """
    Validate uploaded model format
    Returns: model type string or None
    """
    ext = Path(filename).suffix.lower()
    
    format_map = {
        ".pkl": "scikit-learn",
        ".pt": "pytorch",
        ".pth": "pytorch",
        ".h5": "tensorflow",
        ".pb": "tensorflow"
    }
    
    return format_map.get(ext)


async def save_uploaded_model(file, model_id: str, model_dir: str) -> str:
    """Save uploaded model file"""
    try:
        file_path = os.path.join(model_dir, f"{model_id}_{file.filename}")
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Model saved: {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Model save failed: {e}")
        raise


async def validate_data_format(file) -> int:
    """
    Validate uploaded data format
    Returns: number of samples
    """
    try:
        content = await file.read()
        await file.seek(0)  # Reset file pointer
        
        # Try to read as CSV
        df = pd.read_csv(pd.io.common.BytesIO(content))
        
        if df.empty:
            raise ValueError("Empty dataset")
        
        logger.info(f"Data validation passed: {len(df)} samples")
        return len(df)
        
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        raise ValueError(f"Invalid CSV format: {e}")


async def save_uploaded_data(file, model_id: str, data_dir: str) -> str:
    """Save uploaded data file"""
    try:
        file_path = os.path.join(data_dir, f"{model_id}_data.csv")
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Data saved: {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Data save failed: {e}")
        raise


# ============================================================================
# MODEL LOADING & PREDICTION FUNCTIONS
# ============================================================================

def load_model_from_storage(model_id: str, model_dir: str, metadata_dir: str) -> Tuple[Any, Dict]:
    """Load model and metadata from storage"""
    try:
        # Load metadata
        metadata_path = os.path.join(metadata_dir, f"{model_id}_metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        model_path = metadata["model_path"]
        model_format = metadata["model_format"]
        
        # Load model based on format
        if model_format == "scikit-learn":
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        
        elif model_format == "pytorch":
            import torch
            model = torch.load(model_path)
            model.eval()
        
        elif model_format == "tensorflow":
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
        
        else:
            raise ValueError(f"Unsupported model format: {model_format}")
        
        logger.info(f"Model loaded: {model_id}")
        return model, metadata
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise


def make_prediction(model, input_data: Dict[str, Any], metadata: Dict) -> Tuple[Any, Optional[float]]:
    """
    Make prediction using loaded model
    Returns: (prediction, confidence)
    """
    try:
        model_format = metadata["model_format"]
        
        # Convert input to appropriate format
        if model_format == "scikit-learn":
            # Convert dict to array
            feature_order = metadata.get("feature_order", list(input_data.keys()))
            X = np.array([[input_data[f] for f in feature_order]])
            
            prediction = model.predict(X)[0]
            
            # Get confidence if available
            confidence = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                confidence = float(np.max(proba))
        
        elif model_format == "pytorch":
            import torch
            
            feature_order = metadata.get("feature_order", list(input_data.keys()))
            X = torch.tensor([[input_data[f] for f in feature_order]], dtype=torch.float32)
            
            with torch.no_grad():
                output = model(X)
                prediction = output.numpy()[0]
                confidence = None  # Can extract from softmax if classification
        
        elif model_format == "tensorflow":
            feature_order = metadata.get("feature_order", list(input_data.keys()))
            X = np.array([[input_data[f] for f in feature_order]])
            
            prediction = model.predict(X, verbose=0)[0]
            confidence = None
        
        else:
            raise ValueError(f"Unsupported format: {model_format}")
        
        # Convert numpy types to Python types
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()
        elif isinstance(prediction, (np.integer, np.floating)):
            prediction = prediction.item()
        
        logger.info(f"Prediction made: {prediction}, Confidence: {confidence}")
        return prediction, confidence
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


# ============================================================================
# TRUST SCORE CALCULATION FUNCTIONS (Day 1 Placeholder)
# ============================================================================

def calculate_trust_score(
    model: Any,
    input_data: Dict[str, Any],
    metadata: Dict,
    model_id: str,
    data_dir: str
) -> Dict[str, Any]:
    """
    Calculate comprehensive trust score
    (Placeholder for Day 3-4 implementation)
    """
    # For Day 1, return basic structure
    # Will be implemented fully on Days 3-4
    
    trust_breakdown = {
        "confidence_consistency": 85.0,  # Placeholder
        "data_familiarity": 75.0,        # Placeholder
        "agreement_score": 80.0,         # Placeholder
        "explanation_stability": 70.0,   # Placeholder
        "historical_error_rate": 90.0    # Placeholder
    }
    
    # Calculate weighted average
    weights = {
        "confidence_consistency": 0.2,
        "data_familiarity": 0.25,
        "agreement_score": 0.2,
        "explanation_stability": 0.15,
        "historical_error_rate": 0.2
    }
    
    trust_score = sum(trust_breakdown[k] * weights[k] for k in weights.keys())
    
    # Generate explanation
    if trust_score >= 80:
        explanation = "High trust: Model prediction is reliable"
    elif trust_score >= 50:
        explanation = "Medium trust: Use prediction with caution"
    else:
        explanation = "Low trust: Do not rely on this prediction"
    
    return {
        "trust_score": round(trust_score, 2),
        "breakdown": trust_breakdown,
        "explanation": explanation
    }


# ============================================================================
# EXPERIMENT TRACKING FUNCTIONS
# ============================================================================

def log_experiment_metadata(metadata: Dict, experiment_name: str) -> None:
    """Log experiment metadata to MLflow"""
    try:
        import mlflow
        
        mlflow.log_params(metadata)
        logger.info(f"Logged metadata for experiment: {experiment_name}")
        
    except Exception as e:
        logger.error(f"Metadata logging failed: {e}")


def track_model_metrics(metrics: Dict, step: int = None) -> None:
    """Track model performance metrics"""
    try:
        import mlflow
        
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        
        logger.info(f"Logged {len(metrics)} metrics")
        
    except Exception as e:
        logger.error(f"Metrics tracking failed: {e}")


def create_dvc_pipeline(stages: Dict, output_path: Path) -> None:
    """Create DVC pipeline YAML"""
    try:
        import yaml
        
        pipeline = {"stages": stages}
        
        with open(output_path, "w") as f:
            yaml.dump(pipeline, f, default_flow_style=False)
        
        logger.info(f"DVC pipeline created: {output_path}")
        
    except Exception as e:
        logger.error(f"DVC pipeline creation failed: {e}")


# ============================================================================
# PLACEHOLDER FUNCTIONS FOR FUTURE DAYS
# ============================================================================

def process_upload():
    """Placeholder for DVC stage - Day 2"""
    logger.info("Processing upload stage...")


def calculate_all_metrics():
    """Placeholder for DVC stage - Days 3-4"""
    logger.info("Calculating trust metrics...")


def evaluate_system():
    """Placeholder for DVC stage - Day 7"""
    logger.info("Evaluating system performance...")