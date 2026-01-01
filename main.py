"""
ML Trust Score System - Streamlined Interface
Automatically uploads model, makes prediction, calculates trust score, and exits
"""

import os
import json
import mlflow
import dagshub
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from functions import (
    validate_model_format,
    validate_data_format_sync,
    save_model_sync,
    save_data_sync,
    load_model_from_storage,
    make_prediction,
    calculate_trust_score,
    log_to_dvc
)

load_dotenv()

MODEL_DIR = "uploads/models"
DATA_DIR = "uploads/data"
METADATA_DIR = "uploads/metadata"

DIRS = [MODEL_DIR, DATA_DIR, METADATA_DIR, "data/raw", "data/processed", "models", "metrics", "logs"]
for d in DIRS:
    Path(d).mkdir(parents=True, exist_ok=True)

DAGSHUB_REPO = os.getenv("DAGSHUB_REPO", "garvitwork/ml-trust-score")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "")

def init_tracking():
    """Initialize DagHub and MLflow tracking"""
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
            return True
        except Exception as e:
            print(f"⚠ DagHub connection failed: {e}")
            mlflow.set_tracking_uri("file://./mlruns")
            print("⚠ Using local MLflow")
            return False
    else:
        mlflow.set_tracking_uri("file://./mlruns")
        print("⚠ Using local MLflow (DAGSHUB_TOKEN not set)")
        return False

def extract_metadata_from_data(data_path):
    """Automatically extract feature names from CSV"""
    try:
        df = pd.read_csv(data_path)
        feature_order = df.columns.tolist()
        return feature_order
    except Exception as e:
        print(f"⚠ Could not extract features from data: {e}")
        return None

def upload_model():
    """Upload model with automatic metadata extraction"""
    print("\n" + "="*60)
    print("MODEL UPLOAD")
    print("="*60)
    
    model_path = input("\nEnter path to model file (.pkl/.pt/.h5): ").strip()
    if not os.path.exists(model_path):
        print(f"✗ Error: Model file not found at {model_path}")
        return None
    
    model_format = validate_model_format(model_path)
    if not model_format:
        print("✗ Error: Unsupported model format. Use .pkl, .pt, or .h5")
        return None
    print(f"✓ Model format detected: {model_format}")
    
    data_path = input("Enter path to training data CSV: ").strip()
    if not os.path.exists(data_path):
        print(f"✗ Error: Data file not found at {data_path}")
        return None
    
    try:
        data_samples = validate_data_format_sync(data_path)
        print(f"✓ Data validated: {data_samples} samples")
    except Exception as e:
        print(f"✗ Error validating data: {e}")
        return None
    
    # Automatically extract metadata from data
    print("\n⚙ Extracting metadata from data...")
    feature_order = extract_metadata_from_data(data_path)
    if feature_order:
        print(f"✓ Features detected: {', '.join(feature_order)}")
    else:
        print("✗ Could not extract features from data")
        return None
    
    model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=f"upload_{model_id}"):
        try:
            saved_model_path = save_model_sync(model_path, model_id, MODEL_DIR)
            print(f"✓ Model saved: {saved_model_path}")
            
            saved_data_path = save_data_sync(data_path, model_id, DATA_DIR)
            print(f"✓ Data saved: {saved_data_path}")
            
            metadata = {
                "model_id": model_id,
                "model_format": model_format,
                "upload_timestamp": datetime.now().isoformat(),
                "model_path": saved_model_path,
                "data_path": saved_data_path,
                "data_samples": data_samples,
                "feature_order": feature_order
            }
            
            metadata_path = os.path.join(METADATA_DIR, f"{model_id}_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"✓ Metadata saved automatically: {metadata_path}")
            
            mlflow.log_param("model_id", model_id)
            mlflow.log_param("model_format", model_format)
            mlflow.log_param("data_samples", data_samples)
            mlflow.log_artifact(metadata_path)
            mlflow.log_metric("upload_success", 1)
            
            log_to_dvc("upload", {"model_id": model_id, "samples": data_samples})
            
            print(f"\n✓ Upload successful! Model ID: {model_id}")
            return model_id, metadata
            
        except Exception as e:
            mlflow.log_metric("upload_success", 0)
            print(f"✗ Upload failed: {e}")
            return None

def make_predictions_on_all_data(model, metadata, model_id):
    """Make predictions on all data rows with aggregate trust scoring"""
    print("\n" + "="*60)
    print("PREDICTION & TRUST SCORING")
    print("="*60)
    print(f"\nModel ID: {model_id}")
    print(f"Format: {metadata['model_format']}")
    
    feature_order = metadata.get("feature_order")
    print(f"\nFeatures: {', '.join(feature_order)}")
    
    # Load all data for predictions
    print("\n⚙ Loading data for predictions...")
    try:
        data_path = metadata.get("data_path")
        df = pd.read_csv(data_path)
        total_rows = len(df)
        
        print(f"✓ Loaded {total_rows} rows from training data")
        print("\n⚙ Making predictions on all rows...")
        
        all_trust_scores = []
        all_predictions = []
        
        for idx in range(total_rows):
            input_data = {}
            for feature in feature_order:
                input_data[feature] = float(df[feature].iloc[idx])
            
            # Make prediction
            prediction, confidence = make_prediction(model, input_data, metadata)
            all_predictions.append(prediction)
            
            # Calculate trust score for this row
            trust_result = calculate_trust_score(
                model=model,
                input_data=input_data,
                metadata=metadata,
                model_id=model_id,
                data_dir=DATA_DIR
            )
            all_trust_scores.append(trust_result["trust_score"])
            
            if (idx + 1) % 10 == 0 or (idx + 1) == total_rows:
                print(f"  Progress: {idx + 1}/{total_rows} rows processed")
        
        # Calculate aggregate trust score
        avg_trust_score = sum(all_trust_scores) / len(all_trust_scores)
        min_trust_score = min(all_trust_scores)
        max_trust_score = max(all_trust_scores)
        
        print(f"\n✓ Predictions completed on all {total_rows} rows")
        
        # Use the average trust metrics for final breakdown
        input_data = {feature: float(df[feature].iloc[0]) for feature in feature_order}
        trust_result = calculate_trust_score(
            model=model,
            input_data=input_data,
            metadata=metadata,
            model_id=model_id,
            data_dir=DATA_DIR
        )
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    with mlflow.start_run(run_name=f"predict_{model_id[:8]}"):
        try:
            mlflow.log_param("model_id", model_id)
            mlflow.log_param("total_predictions", total_rows)
            
            print("\n" + "="*60)
            print("TRUST SCORE RESULTS")
            print("="*60)
            print(f"\n📊 AGGREGATE TRUST SCORE (across {total_rows} predictions)")
            print(f"\n   Average Trust Score: {avg_trust_score:.2f}")
            print(f"   Minimum Trust Score: {min_trust_score:.2f}")
            print(f"   Maximum Trust Score: {max_trust_score:.2f}")
            
            if avg_trust_score >= 80:
                explanation = "HIGH TRUST (80-100): Model predictions are reliable"
            elif avg_trust_score >= 50:
                explanation = "MEDIUM TRUST (50-80): Use predictions with caution"
            else:
                explanation = "LOW TRUST (0-50): Do NOT rely on these predictions"
            
            print(f"\n   {explanation}")
            print("\n   Sample Breakdown (from representative calculation):")
            for metric, value in trust_result['breakdown'].items():
                print(f"   • {metric}: {value:.2f}")
            print("\n" + "="*60)
            
            mlflow.log_metric("avg_trust_score", avg_trust_score)
            mlflow.log_metric("min_trust_score", min_trust_score)
            mlflow.log_metric("max_trust_score", max_trust_score)
            mlflow.log_metric("total_predictions", total_rows)
            for metric, value in trust_result["breakdown"].items():
                mlflow.log_metric(f"trust_{metric}", value)
            mlflow.log_metric("prediction_success", 1)
            
            log_to_dvc("predict", {
                "model_id": model_id,
                "avg_trust_score": avg_trust_score,
                "total_predictions": total_rows
            })
            
            print("\n✓ Results logged to DVC and MLflow")
            
        except Exception as e:
            mlflow.log_metric("prediction_success", 0)
            print(f"✗ Prediction failed: {e}")

if __name__ == "__main__":
    print("="*60)
    print("ML TRUST SCORE SYSTEM")
    print("="*60)
    
    init_tracking()
    
    # Step 1: Upload model and data
    result = upload_model()
    
    if result:
        model_id, metadata = result
        
        try:
            # Step 2: Load model
            model, metadata = load_model_from_storage(model_id, MODEL_DIR, METADATA_DIR)
            
            # Step 3: Make predictions on all data and calculate trust score
            make_predictions_on_all_data(model, metadata, model_id)
            
        except Exception as e:
            print(f"✗ Error: {e}")
    else:
        print("\n✗ Model upload failed.")
    
    # Step 4: Exit
    print("\n✓ Thank you for using ML Trust Score System!")
    print("="*60)