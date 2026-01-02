# ML Trust Score System

A system to evaluate trustworthiness of ML model predictions using multiple metrics with DVC and DagHub tracking.

## 🏗️ Project Structure

```
ml-trust-score/
├── main.py              # Interactive terminal interface (PRIMARY)
├── api.py               # FastAPI server (for deployment)
├── functions.py         # Core logic and metrics
├── requirements.txt     # Dependencies
├── docker-compose.yml   # Docker setup
├── .env                 # Environment variables
├── uploads/
│   ├── models/         # Uploaded model files
│   ├── data/           # Training data CSVs
│   └── metadata/       # Model metadata JSON
└── metrics/            # DVC-tracked metrics
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Environment
Create `.env` file:
```bash
DAGSHUB_REPO=garvitwork/ml-trust-score
DAGSHUB_TOKEN=your-dagshub-token
```

### 3. Run Interactive Terminal
```bash
python main.py
```

This will:
- Initialize DVC and DagHub
- Show interactive menu
- Prompt for model/data paths
- Track everything with MLflow + DVC

## 📝 Using main.py (Interactive Mode)

### Menu Options:

```
1. Upload model and data
2. Make prediction with trust scoring
3. List uploaded models
4. Exit
```

### Example Session:

```bash
$ python main.py

================================================================
ML TRUST SCORE SYSTEM - INTERACTIVE MODE
================================================================
✓ DagHub connected: garvitwork/ml-trust-score

================================================================
ML TRUST SCORE SYSTEM
================================================================

Options:
1. Upload model and data
2. Make prediction with trust scoring
3. List uploaded models
4. Exit

Select option (1-4): 1

================================================================
MODEL UPLOAD
================================================================

Enter path to model file (.pkl/.pt/.h5): models/my_model.pkl
✓ Model format detected: scikit-learn

Enter path to training data CSV: data/training_data.csv
✓ Data validated: 1000 samples

Enter metadata (or press Enter to skip):
  Feature order (comma-separated, e.g., 'age,income,score'): age,income,score

✓ Model saved: uploads/models/model_20240101_120000_my_model.pkl
✓ Data saved: uploads/data/model_20240101_120000_data.csv
✓ Metadata saved: uploads/metadata/model_20240101_120000_metadata.json

✓ Upload successful! Model ID: model_20240101_120000

Make a prediction now? (y/n): y

================================================================
PREDICTION & TRUST SCORING
================================================================

Required features: age, income, score

  age: 35
  income: 75000
  score: 0.8

✓ Prediction: 1
  Confidence: 87.32%

📊 TRUST SCORE: 78.5
   MEDIUM TRUST (50-80): Use prediction with caution

   Breakdown:
   • confidence_consistency: 85.23
   • data_familiarity: 72.10
   • agreement_score: 75.00
   • explanation_stability: 100.00
   • historical_error_rate: 85.00

✓ Prediction logged to DVC and MLflow
```

## 🌐 Using api.py (FastAPI Server)

### Start the API Server:
```bash
# Option 1: Direct
python api.py

# Option 2: With uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Option 3: Docker
docker-compose up --build
```

### API Endpoints:

**1. Health Check**
```bash
GET http://localhost:8000/
```

**2. Upload Model**
```bash
POST http://localhost:8000/upload
Content-Type: multipart/form-data

Files:
- model_file: my_model.pkl
- data_file: training_data.csv
- metadata: metadata.json
```

Example metadata.json:
```json
{
  "feature_order": ["age", "income", "score"],
  "training_params": {
    "algorithm": "RandomForest",
    "n_estimators": 100
  }
}
```

**3. Make Prediction**
```bash
POST http://localhost:8000/predict
Content-Type: application/json

{
  "model_id": "model_20240101_120000",
  "input_data": {
    "age": 35,
    "income": 75000,
    "score": 0.8
  }
}
```

**4. List Models**
```bash
GET http://localhost:8000/models
```

### API Documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📊 Trust Metrics Explained

### 1. Confidence Consistency (20%)
Tests prediction stability by adding small noise to input.
- **High score**: Model is robust to small variations

### 2. Data Familiarity (25%)
Measures how similar input is to training data.
- **High score**: Input is within known data distribution

### 3. Agreement Score (20%)
Compares predictions across multiple models.
- **Current**: Placeholder at 75.0

### 4. Explanation Stability (15%)
Checks if model gives consistent predictions.
- **High score**: Deterministic behavior

### 5. Historical Error Rate (20%)
Tracks past prediction accuracy.
- **Current**: Placeholder at 85.0

### Final Trust Score:
- **80-100**: HIGH TRUST - Reliable prediction
- **50-80**: MEDIUM TRUST - Use with caution
- **0-50**: LOW TRUST - Do not rely on this

## 🐳 Docker Deployment

### Build and Run:
```bash
docker-compose up --build
```

### Environment Variables:
Set in `.env` file or docker-compose.yml:
```yaml
environment:
  - DAGSHUB_REPO=your-username/ml-trust-score
  - DAGSHUB_TOKEN=your-token
```

### Volumes:
```yaml
volumes:
  - ./uploads:/app/uploads    # Model storage
  - ./metrics:/app/metrics    # DVC metrics
```

## 📈 DVC & DagHub Tracking

### What's Tracked:
- Model uploads (timestamp, format, samples)
- Predictions (trust scores, breakdown)
- MLflow experiments (parameters, metrics)

### View Metrics:
```bash
# Local
cat metrics/trust_scores.json

# DagHub (if configured)
# Visit: https://dagshub.com/your-username/ml-trust-score
```

## 🔧 Supported Model Formats

| Format | Extension | Framework |
|--------|-----------|-----------|
| Pickle | .pkl | scikit-learn |
| PyTorch | .pt | PyTorch |
| TensorFlow | .h5 | Keras/TensorFlow |

## 📁 Data Format

Training data must be CSV with:
- Feature columns matching `feature_order` in metadata
- Target column (optional, not used for prediction)

Example:
```csv
age,income,score,target
35,75000,0.8,1
42,85000,0.7,0
28,60000,0.9,1
```

## 🛠️ Troubleshooting

### "Model not found" error:
```bash
# List available models
python main.py
# Select option 3
```

### Port 8000 already in use:
```bash
# Kill existing process
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn api:app --port 8001
```

### DagHub connection failed:
```bash
# Check .env file
cat .env

# Test connection
python -c "import dagshub; print(dagshub.__version__)"
```

## 📝 Example Workflow

### 1. Prepare Your Files:
```bash
models/
  └── my_model.pkl
data/
  └── training_data.csv
```

### 2. Create metadata.json:
```json
{
  "feature_order": ["feature1", "feature2", "feature3"]
}
```

### 3. Run main.py:
```bash
python main.py
# Choose option 1
# Enter file paths when prompted
```

### 4. Make Predictions:
```bash
# Still in main.py
# Choose option 2
# Enter model_id and feature values
```

## 🚀 Deployment to Render

### 1. Push to GitHub:
```bash
git add .
git commit -m "ML Trust Score System"
git push origin main
```

### 2. Create Render Service:
- Go to render.com
- Create new Web Service
- Connect your GitHub repo
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`

### 3. Set Environment Variables:
```
DAGSHUB_REPO=your-username/ml-trust-score
DAGSHUB_TOKEN=your-token
```

## 📚 Next Steps

- **Day 3**: Implement multi-model agreement
- **Day 4**: Add historical error tracking
- **Day 5**: Build web interface
- **Day 6**: Integration testing
- **Day 7**: Documentation & evaluation

## 🤝 Contributing

This is a prototype system. Improvements welcome!

## 📄 License

MIT License
