# MLOps Credit Risk Pipeline

An end-to-end MLOps pipeline that trains, evaluates, deploys, and monitors a credit default risk model — demonstrating production ML engineering practices.

## Architecture

```
Raw Data → Feature Engineering → Model Training → MLflow Registry
                                                         │
                                               Quality Gate (evaluate.py)
                                                         │
                                               Production Model
                                                         │
                                            FastAPI Inference Server
                                                         │
                                            Prediction Logging → Drift Monitoring
                                                                         │
                                                               Streamlit Dashboard
```

## Project Structure

```
mlops-pipeline/
├── src/
│   ├── ingest.py          # Data loading & validation
│   ├── features.py        # Feature engineering & preprocessing
│   ├── train.py           # Training pipeline + MLflow logging
│   ├── evaluate.py        # Model evaluation & registry promotion
│   └── predict.py         # FastAPI inference server
├── monitoring/
│   ├── drift_report.py    # Evidently AI drift detection
│   └── dashboard.py       # Streamlit ops dashboard
├── tests/
│   └── test_features.py   # Unit tests
├── data/raw/              # Place credit.csv here
├── artifacts/             # Generated model artifacts
├── .github/workflows/
│   └── ci_cd.yml          # GitHub Actions pipeline
├── Dockerfile
└── requirements.txt
```

## Quick Start

### 1. Clone and set up environment

```bash
git clone <your-repo-url>
cd mlops-pipeline
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Download the dataset

Download from Kaggle: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

Place the CSV at: `data/raw/credit.csv`

### 3. Run the full pipeline

```bash
# Start MLflow UI (keep this terminal open)
mlflow ui

# In a new terminal — train the model
python src/train.py

# Evaluate and promote to Production
python src/evaluate.py

# Start the inference API
python src/predict.py

# In another terminal — view the dashboard
streamlit run monitoring/dashboard.py
```

### 4. Test the API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2,
    "MARRIAGE": 1, "AGE": 30,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": -1, "PAY_6": -1,
    "BILL_AMT1": 10000, "BILL_AMT2": 9500, "BILL_AMT3": 9000,
    "BILL_AMT4": 8500, "BILL_AMT5": 8000, "BILL_AMT6": 7500,
    "PAY_AMT1": 5000, "PAY_AMT2": 4500, "PAY_AMT3": 4000,
    "PAY_AMT4": 3500, "PAY_AMT5": 3000, "PAY_AMT6": 2500
  }'
```

### 5. Run unit tests

```bash
pytest tests/ -v
```

### 6. Check drift

```bash
python monitoring/drift_report.py
```

## Model Details

- **Algorithm**: Stacking ensemble (LightGBM + XGBoost + Random Forest) with Logistic Regression meta-learner
- **Calibration**: Isotonic regression calibration for reliable probability outputs
- **Hyperparameter tuning**: Optuna with 30 trials
- **Validation**: 5-fold stratified cross-validation
- **Primary metric**: ROC-AUC
- **Dataset**: UCI Credit Card Default (30,000 customers)

## Quality Gates

The evaluate.py script enforces two gates before any model reaches Production:

1. **Absolute floor**: AUC must be ≥ 0.72
2. **Improvement gate**: New model must beat current Production by ≥ 0.003 AUC

If either gate fails, the CI/CD pipeline exits with code 1 and no deployment occurs.

## Docker

```bash
# Build
docker build -t mlops-pipeline .

# Train
docker run mlops-pipeline

# Evaluate
docker run mlops-pipeline python src/evaluate.py

# Serve API
docker run -p 8000:8000 mlops-pipeline python src/predict.py
```

## URLs when running locally

| Service | URL |
|---|---|
| MLflow Tracking UI | http://localhost:5000 |
| FastAPI Inference | http://localhost:8000 |
| FastAPI Docs | http://localhost:8000/docs |
| Streamlit Dashboard | http://localhost:8501 |