"""
Main Pipeline Runner with DVC and DagHub Tracking
Orchestrates the ML Trust Score System workflow
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
import dagshub
import mlflow

from functions import (
    initialize_dvc,
    setup_dagshub_connection,
    create_pipeline_structure,
    log_experiment_metadata,
    track_model_metrics,
    create_dvc_pipeline
)

from dotenv import load_dotenv
load_dotenv()


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_NAME = "ml-trust-score-system"
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO", "garvitwork/ml-trust-score")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "f43d18eaef53b8269dd37f6434a8612b4faa6c8b")

# Directory structure
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "metrics"
LOGS_DIR = BASE_DIR / "logs"

# Create directories
for directory in [DATA_DIR, MODELS_DIR, METRICS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class MLTrustPipeline:
    """Main pipeline orchestrator with MLOps tracking"""
    
    def __init__(self):
        self.experiment_id = None
        self.run_id = None
        self.config = self.load_config()
        
    def load_config(self):
        """Load pipeline configuration"""
        config_path = BASE_DIR / "config.yaml"
        
        if not config_path.exists():
            # Create default config
            default_config = {
                "pipeline": {
                    "name": PROJECT_NAME,
                    "version": "1.0.0"
                },
                "supported_formats": {
                    "models": [".pkl", ".pt", ".h5"],
                    "data": [".csv"]
                },
                "trust_metrics": {
                    "confidence_consistency": {"weight": 0.2, "enabled": True},
                    "data_familiarity": {"weight": 0.25, "enabled": True},
                    "agreement_score": {"weight": 0.2, "enabled": True},
                    "explanation_stability": {"weight": 0.15, "enabled": True},
                    "historical_error_rate": {"weight": 0.2, "enabled": True}
                },
                "thresholds": {
                    "high_trust": 80,
                    "medium_trust": 50,
                    "low_trust": 0
                }
            }
            
            with open(config_path, "w") as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            logger.info(f"Created default config at {config_path}")
            return default_config
        
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    def initialize_tracking(self):
        """Initialize DVC and DagHub tracking"""
        try:
            logger.info("Initializing DVC...")
            initialize_dvc(BASE_DIR)
            
            logger.info("Setting up DagHub connection...")
            if DAGSHUB_TOKEN:
                dagshub.init(repo_owner=DAGSHUB_REPO.split('/')[0],
                            repo_name=DAGSHUB_REPO.split('/')[1],
                            mlflow=True)
                
                mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_REPO}.mlflow")
                logger.info(f"Connected to DagHub: {DAGSHUB_REPO}")
            else:
                logger.warning("DAGSHUB_TOKEN not set. Using local MLflow tracking.")
                mlflow.set_tracking_uri(f"file://{BASE_DIR / 'mlruns'}")
            
            # Set experiment
            mlflow.set_experiment(PROJECT_NAME)
            
            return True
            
        except Exception as e:
            logger.error(f"Tracking initialization failed: {e}")
            return False
    
    def setup_dvc_pipeline(self):
        """Create DVC pipeline configuration"""
        try:
            logger.info("Creating DVC pipeline...")
            
            dvc_yaml = {
                "stages": {
                    "upload_model": {
                        "cmd": "python -c 'from functions import process_upload; process_upload()'",
                        "deps": ["data/raw"],
                        "outs": ["models/", "data/processed/"]
                    },
                    "calculate_trust": {
                        "cmd": "python -c 'from functions import calculate_all_metrics; calculate_all_metrics()'",
                        "deps": ["models/", "data/processed/"],
                        "metrics": ["metrics/trust_scores.json"]
                    },
                    "evaluate": {
                        "cmd": "python -c 'from functions import evaluate_system; evaluate_system()'",
                        "deps": ["metrics/trust_scores.json"],
                        "plots": ["metrics/trust_distribution.csv"]
                    }
                }
            }
            
            dvc_yaml_path = BASE_DIR / "dvc.yaml"
            with open(dvc_yaml_path, "w") as f:
                yaml.dump(dvc_yaml, f, default_flow_style=False)
            
            logger.info(f"DVC pipeline created at {dvc_yaml_path}")
            
            # Create .dvcignore
            dvcignore_path = BASE_DIR / ".dvcignore"
            with open(dvcignore_path, "w") as f:
                f.write("# DVC ignore file\n")
                f.write("*.pyc\n")
                f.write("__pycache__/\n")
                f.write(".env\n")
                f.write("mlruns/\n")
            
            return True
            
        except Exception as e:
            logger.error(f"DVC pipeline setup failed: {e}")
            return False
    
    def run_experiment(self, experiment_name: str = None):
        """Run a tracked experiment"""
        try:
            exp_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with mlflow.start_run(run_name=exp_name) as run:
                self.run_id = run.info.run_id
                logger.info(f"Started MLflow run: {self.run_id}")
                
                # Log configuration
                mlflow.log_params({
                    "pipeline_version": self.config["pipeline"]["version"],
                    "project_name": PROJECT_NAME
                })
                
                # Log trust metric weights
                for metric, config in self.config["trust_metrics"].items():
                    mlflow.log_param(f"weight_{metric}", config["weight"])
                    mlflow.log_param(f"enabled_{metric}", config["enabled"])
                
                # Log thresholds
                mlflow.log_params(self.config["thresholds"])
                
                # Log config file as artifact
                mlflow.log_artifact(str(BASE_DIR / "config.yaml"))
                
                logger.info(f"Experiment '{exp_name}' configuration logged")
                
                return self.run_id
                
        except Exception as e:
            logger.error(f"Experiment run failed: {e}")
            return None
    
    def log_system_metrics(self, metrics: dict):
        """Log system performance metrics"""
        try:
            with mlflow.start_run(run_id=self.run_id):
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)
                
                logger.info(f"Logged {len(metrics)} metrics to MLflow")
                
        except Exception as e:
            logger.error(f"Metrics logging failed: {e}")
    
    def save_pipeline_state(self):
        """Save current pipeline state"""
        try:
            state = {
                "timestamp": datetime.now().isoformat(),
                "run_id": self.run_id,
                "config": self.config,
                "directories": {
                    "data": str(DATA_DIR),
                    "models": str(MODELS_DIR),
                    "metrics": str(METRICS_DIR)
                }
            }
            
            state_path = BASE_DIR / "pipeline_state.json"
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Pipeline state saved to {state_path}")
            
        except Exception as e:
            logger.error(f"State save failed: {e}")


def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("ML Trust Score System - Pipeline Runner")
    logger.info("=" * 60)
    
    # Initialize pipeline
    pipeline = MLTrustPipeline()
    
    # Step 1: Initialize tracking
    logger.info("\n[1/4] Initializing DVC and DagHub tracking...")
    if not pipeline.initialize_tracking():
        logger.error("Tracking initialization failed. Exiting.")
        sys.exit(1)
    
    # Step 2: Setup DVC pipeline
    logger.info("\n[2/4] Setting up DVC pipeline...")
    if not pipeline.setup_dvc_pipeline():
        logger.error("DVC pipeline setup failed. Exiting.")
        sys.exit(1)
    
    # Step 3: Run experiment
    logger.info("\n[3/4] Starting experiment tracking...")
    run_id = pipeline.run_experiment("initial_setup")
    if not run_id:
        logger.error("Experiment tracking failed. Exiting.")
        sys.exit(1)
    
    # Step 4: Save state
    logger.info("\n[4/4] Saving pipeline state...")
    pipeline.save_pipeline_state()
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Pipeline initialization complete!")
    logger.info(f"✓ MLflow Run ID: {run_id}")
    logger.info(f"✓ Config: {BASE_DIR / 'config.yaml'}")
    logger.info(f"✓ DVC Pipeline: {BASE_DIR / 'dvc.yaml'}")
    logger.info("=" * 60)
    
    logger.info("\nNext steps:")
    logger.info("1. Start FastAPI server: python app.py")
    logger.info("2. Upload models via /upload endpoint")
    logger.info("3. Run predictions via /predict endpoint")
    logger.info("4. Track experiments: mlflow ui")
    logger.info("5. Version data: dvc add data/")


if __name__ == "__main__":
    main()