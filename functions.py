"""
Core Functions - Model handling, trust metrics, and DVC logging
Includes both sync (for main.py) and async (for api.py) functions
"""

import os
import json
import pickle
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# ============================================================================
# FILE HANDLING - SYNC VERSIONS (for main.py)
# ============================================================================

def validate_model_format(filename: str) -> Optional[str]:
    """Validate model file format"""
    ext = Path(filename).suffix.lower()
    formats = {".pkl": "scikit-learn", ".pt": "pytorch", ".h5": "tensorflow"}
    return formats.get(ext)

def validate_data_format_sync(file_path: str) -> int:
    """Validate CSV data (sync version)"""
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError("Empty dataset")
    return len(df)

def save_model_sync(source_path: str, model_id: str, model_dir: str) -> str:
    """Save model file (sync version)"""
    os.makedirs(model_dir, exist_ok=True)
    filename = os.path.basename(source_path)
    dest_path = os.path.join(model_dir, f"{model_id}_{filename}")
    shutil.copy2(source_path, dest_path)
    return dest_path

def save_data_sync(source_path: str, model_id: str, data_dir: str) -> str:
    """Save data file (sync version)"""
    os.makedirs(data_dir, exist_ok=True)
    dest_path = os.path.join(data_dir, f"{model_id}_data.csv")
    shutil.copy2(source_path, dest_path)
    return dest_path

# ============================================================================
# FILE HANDLING - ASYNC VERSIONS (for api.py)
# ============================================================================

async def save_uploaded_model(file, model_id: str, model_dir: str) -> str:
    """Save model file (async version)"""
    os.makedirs(model_dir, exist_ok=True)
    file_path = os.path.join(model_dir, f"{model_id}_{file.filename}")
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    return file_path

async def validate_data_format(file) -> int:
    """Validate CSV data (async version)"""
    content = await file.read()
    await file.seek(0)
    df = pd.read_csv(pd.io.common.BytesIO(content))
    if df.empty:
        raise ValueError("Empty dataset")
    return len(df)

async def save_uploaded_data(file, model_id: str, data_dir: str) -> str:
    """Save data file (async version)"""
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{model_id}_data.csv")
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    return file_path

# ============================================================================
# MODEL OPERATIONS (used by both main.py and api.py)
# ============================================================================

def load_model_from_storage(model_id: str, model_dir: str, metadata_dir: str) -> Tuple[Any, Dict]:
    """Load model and metadata"""
    metadata_path = os.path.join(metadata_dir, f"{model_id}_metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    model_path = metadata["model_path"]
    model_format = metadata["model_format"]
    
    if model_format == "scikit-learn":
        # Try joblib first (more reliable for sklearn)
        try:
            import joblib
            model = joblib.load(model_path)
        except:
            # Fall back to pickle
            with open(model_path, "rb") as f:
                model = pickle.load(f)
                
    elif model_format == "pytorch":
        import torch
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
    elif model_format == "tensorflow":
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
    else:
        raise ValueError(f"Unsupported format: {model_format}")
    
    return model, metadata

def make_prediction(model, input_data: Dict[str, Any], metadata: Dict) -> Tuple[Any, Optional[float]]:
    """Make prediction"""
    model_format = metadata["model_format"]
    feature_order = metadata.get("feature_order", list(input_data.keys()))
    
    # Create DataFrame with proper column names for sklearn
    if model_format == "scikit-learn":
        import pandas as pd
        X = pd.DataFrame([[input_data[f] for f in feature_order]], columns=feature_order)
        
        prediction = model.predict(X)[0]
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            confidence = float(np.max(proba))
    
    elif model_format == "pytorch":
        import torch
        X = np.array([[input_data[f] for f in feature_order]])
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            output = model(X_tensor)
            prediction = output.numpy()[0]
            confidence = None
    
    elif model_format == "tensorflow":
        X = np.array([[input_data[f] for f in feature_order]])
        prediction = model.predict(X, verbose=0)[0]
        confidence = None
    
    # Convert to Python types
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()
    elif isinstance(prediction, (np.integer, np.floating)):
        prediction = prediction.item()
    
    return prediction, confidence

# ============================================================================
# TRUST METRICS
# ============================================================================

def calculate_confidence_consistency(model, input_data: Dict, metadata: Dict) -> float:
    """Metric 1: Confidence Consistency - Test stability with noise"""
    try:
        import pandas as pd
        feature_order = metadata.get("feature_order", list(input_data.keys()))
        
        # Original prediction
        original_pred, _ = make_prediction(model, input_data, metadata)
        
        # Add small noise
        X = np.array([[input_data[f] for f in feature_order]])
        noise = np.random.normal(0, 0.05, X.shape)
        X_noisy = X + noise
        noisy_input = {f: float(X_noisy[0][i]) for i, f in enumerate(feature_order)}
        noisy_pred, _ = make_prediction(model, noisy_input, metadata)
        
        # Calculate consistency
        if isinstance(original_pred, (list, np.ndarray)):
            diff = np.linalg.norm(np.array(original_pred) - np.array(noisy_pred))
        else:
            diff = abs(float(original_pred) - float(noisy_pred))
        
        consistency = max(0, 100 - (diff * 100))
        return consistency
    except:
        return 50.0

def calculate_data_familiarity(input_data: Dict, data_path: str, metadata: Dict) -> float:
    """Metric 2: Data Familiarity - Check similarity to training data"""
    try:
        df = pd.read_csv(data_path)
        feature_order = metadata.get("feature_order", list(input_data.keys()))
        
        input_vector = np.array([input_data[f] for f in feature_order])
        train_vectors = df[feature_order].values
        
        # Calculate distances
        distances = np.linalg.norm(train_vectors - input_vector, axis=1)
        min_dist = np.min(distances)
        avg_dist = np.mean(distances)
        
        # Convert to familiarity score
        familiarity = max(0, 100 - (min_dist / avg_dist * 100))
        return familiarity
    except:
        return 50.0

def calculate_agreement_score(input_data: Dict) -> float:
    """Metric 3: Agreement Score - Placeholder for multi-model agreement"""
    # Would need multiple models to compare
    return 75.0

def calculate_explanation_stability(model, input_data: Dict, metadata: Dict) -> float:
    """Metric 4: Explanation Stability - Check prediction consistency"""
    try:
        pred1, _ = make_prediction(model, input_data, metadata)
        pred2, _ = make_prediction(model, input_data, metadata)
        
        if isinstance(pred1, (list, np.ndarray)):
            stable = np.allclose(pred1, pred2)
        else:
            stable = (pred1 == pred2)
        
        return 100.0 if stable else 50.0
    except:
        return 50.0

def calculate_historical_error_rate() -> float:
    """Metric 5: Historical Error Rate - Placeholder for past performance"""
    # Would need historical predictions
    return 85.0

def calculate_trust_score(
    model: Any,
    input_data: Dict[str, Any],
    metadata: Dict,
    model_id: str,
    data_dir: str
) -> Dict[str, Any]:
    """Calculate comprehensive trust score using all 5 metrics"""
    
    data_path = os.path.join(data_dir, f"{model_id}_data.csv")
    
    # Calculate all metrics
    breakdown = {
        "confidence_consistency": calculate_confidence_consistency(model, input_data, metadata),
        "data_familiarity": calculate_data_familiarity(input_data, data_path, metadata),
        "agreement_score": calculate_agreement_score(input_data),
        "explanation_stability": calculate_explanation_stability(model, input_data, metadata),
        "historical_error_rate": calculate_historical_error_rate()
    }
    
    # Weighted average
    weights = {
        "confidence_consistency": 0.2,
        "data_familiarity": 0.25,
        "agreement_score": 0.2,
        "explanation_stability": 0.15,
        "historical_error_rate": 0.2
    }
    
    trust_score = sum(breakdown[k] * weights[k] for k in weights.keys())
    
    # Generate explanation based on thresholds
    if trust_score >= 80:
        explanation = "HIGH TRUST (80-100): Model prediction is reliable"
    elif trust_score >= 50:
        explanation = "MEDIUM TRUST (50-80): Use prediction with caution"
    else:
        explanation = "LOW TRUST (0-50): Do NOT rely on this prediction"
    
    return {
        "trust_score": round(trust_score, 2),
        "breakdown": {k: round(v, 2) for k, v in breakdown.items()},
        "explanation": explanation
    }

# ============================================================================
# DVC TRACKING
# ============================================================================

def log_to_dvc(stage: str, data: Dict):
    """Log metrics to DVC-tracked JSON file"""
    try:
        metrics_file = "metrics/trust_scores.json"
        
        # Load existing metrics
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
        else:
            metrics = {"uploads": [], "predictions": []}
        
        # Add new entry
        entry = {
            "timestamp": pd.Timestamp.now().isoformat(),
            **data
        }
        
        if stage == "upload":
            metrics["uploads"].append(entry)
        elif stage == "predict":
            metrics["predictions"].append(entry)
        
        # Save metrics
        os.makedirs("metrics", exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
    except Exception as e:
        print(f"⚠ DVC logging failed: {e}")