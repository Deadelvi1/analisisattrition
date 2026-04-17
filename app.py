from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import json
from pathlib import Path

app = Flask(__name__)

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "model"
MODEL_PATH = MODEL_DIR / "rf_model_tuning_latest.pkl"
MODEL_ALT_PATH = MODEL_DIR / "rf_model_latest.pkl"
MODEL_FALLBACK_PATH = MODEL_DIR / "rf_model.pkl"
PERF_PATH = MODEL_DIR / "performance.json"
MLFLOW_TRACKING_URI = "https://dagshub.com/deadelvina9/attrition-mlops.mlflow"
MODEL_REGISTRY_URI = "models:/rf_model_tuning/2"
model = None
MODEL_READY = False
mlflow = None

performance = {
    "test": {"accuracy": 83.96, "f1": 52.78, "precision": 52.78, "recall": 52.78},
    "val": {"accuracy": 83.96},
    "overfit_gap": 0.0,
    "cv_f1_mean": 51.4,
    "cv_f1_std": 1.8
}


def load_remote_model():
    global model, MODEL_READY, mlflow
    if mlflow is None:
        try:
            import mlflow
            mlflow = mlflow
        except ImportError:
            print("Remote model load skipped because mlflow is not installed")
            return False

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        remote_model = mlflow.sklearn.load_model(MODEL_REGISTRY_URI)
        if remote_model is not None:
            model = remote_model
            MODEL_READY = True
            os.makedirs(ROOT_DIR / "model", exist_ok=True)
            joblib.dump(model, MODEL_ALT_PATH)
            print(f"Loaded remote model and saved local copy to {MODEL_ALT_PATH}")
            return True
    except Exception as e:
        print(f"Remote model load failed: {e}")
    return False


def load_model():
    global model, MODEL_READY, MODEL_PATH

    candidate_paths = [MODEL_PATH, MODEL_ALT_PATH, MODEL_FALLBACK_PATH]
    resolved_candidates = []

    for candidate in candidate_paths:
        if candidate.exists():
            resolved_candidates.append(candidate)

    if not resolved_candidates and MODEL_DIR.exists():
        resolved_candidates = sorted(MODEL_DIR.glob("*.pkl"))

    print(f"Model directory: {MODEL_DIR}")
    print(f"Resolved model candidates: {resolved_candidates}")

    for candidate in resolved_candidates:
        try:
            model = joblib.load(candidate)
            MODEL_READY = True
            MODEL_PATH = candidate
            print(f"Loaded local model from {candidate}")
            return
        except Exception as e:
            print(f"Local model load failed for {candidate}: {e}")

    print("No valid local model found. Attempting remote model load from Dagshub...")
    if load_remote_model():
        return

    MODEL_READY = False


load_model()

try:
    if PERF_PATH.exists():
        with open(PERF_PATH, "r") as f:
            performance = json.load(f)
except Exception:
    pass


@app.route('/')
def home():
    global performance
    try:
        with open(PERF_PATH, "r") as f:
            performance = json.load(f)
    except:
        pass
    perf = performance
    return render_template('home.html', perf=perf)


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global performance
    result = None
    note = None

    if request.method == 'GET' and not MODEL_READY:
        note = (
            "Model belum siap. Pastikan file `model/rf_model_tuning_latest.pkl` dan `model/performance.json` "
            "tersedia di repository, lalu push ulang ke Railway."
        )

    if request.method == 'POST':
        if not MODEL_READY:
            note = (
                "Model belum siap. Pastikan file `model/rf_model_tuning_latest.pkl` dan `model/performance.json` "
                "tersedia di repository, lalu push ulang ke Railway."
            )
            return render_template("form_prediction.html", result=None, note=note, model_ready=MODEL_READY)

        try:
            try:
                with open(PERF_PATH, "r") as f:
                    performance = json.load(f)
            except:
                pass
            
            raw_data = request.form.to_dict()
            df = pd.DataFrame([raw_data])
            
            required_features = list(model.feature_names_in_)
            
            for col in required_features:
                if col not in df.columns:
                    df[col] = '0'
             
            df = df[required_features]
            
            numeric_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 
                           'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
                           'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
                           'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
                           'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
                           'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                           'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
                           'EmployeeId']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            pred = model.predict(df)[0]

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(df)[0][1]
            else:
                prob = 0.0

            status_model = "Resign" if pred == 1 else "Bertahan"

            if pred == 0:
                confidence_pct = round((1 - prob) * 100, 2)
            else:
                confidence_pct = round(prob * 100, 2)

            explanation = []

            if pred == 1 and prob < 0.4:
                explanation.append("Confidence rendah")

            if not explanation:
                explanation.append("Konsisten")

            if confidence_pct >= 70:
                confidence_class = "success"
            elif confidence_pct >= 40:
                confidence_class = "warning"
            else:
                confidence_class = "danger"

            result = {
                "status": status_model,
                "confidence": confidence_pct,
                "confidence_class": confidence_class,
                "explanation": " | ".join(explanation),
                "performance": performance
            }

        except Exception as e:
            note = f"Error: {str(e)}"

    return render_template("form_prediction.html", result=result, note=note, model_ready=MODEL_READY)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)