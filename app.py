from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import json
import mlflow
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

performance = {
    "test": {"accuracy": 83.96, "f1": 52.78, "precision": 52.78, "recall": 52.78},
    "val": {"accuracy": 83.96},
    "overfit_gap": 0.0,
    "cv_f1_mean": 51.4,
    "cv_f1_std": 1.8
}


def load_remote_model():
    global model, MODEL_READY
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


def compute_rule_risk(raw):
    def num(key, default=0):
        try:
            return float(raw.get(key, default))
        except:
            return default

    score = 0

    if num('EnvironmentSatisfaction', 4) <= 2:
        score += 2
    if num('JobSatisfaction', 4) <= 2:
        score += 2
    if raw.get('OverTime', '').lower() in ['yes', 'y', 'true', '1']:
        score += 2
    if num('WorkLifeBalance', 4) <= 2:
        score += 1
    if num('YearsSinceLastPromotion', 0) >= 3:
        score += 1
    if num('YearsAtCompany', 0) <= 1:
        score += 1
    if num('DistanceFromHome', 0) > 20:
        score += 1
    if num('NumCompaniesWorked', 0) >= 5:
        score += 1
    if num('MonthlyIncome', 0) < 4500:
        score += 1

    return min(1.0, score / 12) * 100


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

            pred = model.predict(df)[0]

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(df)[0][1]
            else:
                prob = 0.0

            risk_score = compute_rule_risk(raw_data)

            status_model = "Resign" if pred == 1 else "Bertahan"
            status_rule = "Resign" if risk_score >= 55 else "Bertahan"

            final_status = status_model

            explanation = []

            if status_model != status_rule:
                explanation.append(f"Model: {status_model}, Rule: {status_rule}")

            if risk_score >= 70 and status_model == "Bertahan":
                explanation.append("Risiko tinggi")

            if pred == 1 and prob < 0.4:
                explanation.append("Confidence rendah")

            if not explanation:
                explanation.append("Konsisten")

            confidence_pct = round(prob * 100, 2)

            if confidence_pct >= 70:
                confidence_class = "success"
            elif confidence_pct >= 40:
                confidence_class = "warning"
            else:
                confidence_class = "danger"

            result = {
                "status": final_status,
                "confidence": confidence_pct,
                "confidence_class": confidence_class,
                "rule_risk": round(risk_score, 2),
                "model_status": status_model,
                "rule_status": status_rule,
                "explanation": " | ".join(explanation),
                "performance": performance
            }

        except Exception as e:
            note = f"Error: {str(e)}"

    return render_template("form_prediction.html", result=result, note=note, model_ready=MODEL_READY)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not MODEL_READY:
        return jsonify({"error": "Model belum siap"})

    try:
        data = request.json
        df = pd.DataFrame([data])
        pred = model.predict(df)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(df)[0][1]
        else:
            prob = 0.0

        return jsonify({
            "prediction": int(pred),
            "confidence": round(prob * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)