from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

def load_all_resources():
    try:
        model = joblib.load('model/attrition_model.pkl')
        scaler = joblib.load('model/scaler.pkl')
        encoders = joblib.load('model/encoders.pkl')
        cols = joblib.load('model/model_columns.pkl')
        perf = joblib.load('model/performance.pkl')
        return model, scaler, encoders, cols, perf
    except Exception:
        return None, None, None, None, {
            'train': {'accuracy': 0, 'f1': 0},
            'val': {'accuracy': 0, 'f1': 0},
            'test': {'accuracy': 0, 'f1': 0},
            'overfit_gap': 0,
            'overfit_detected': False,
            'cv_f1_mean': 0,
            'cv_f1_std': 0
        }


def safe_label_transform(value, le):
    value_str = str(value).strip()
    if value_str in le.classes_:
        return le.transform([value_str])[0]
    for cls in le.classes_:
        if str(cls).strip().lower() == value_str.lower():
            return le.transform([cls])[0]
    return 0


def compute_rule_risk(raw):
    def num(key, default=0):
        try:
            return float(raw.get(key, default))
        except Exception:
            return default

    score = 0
    if num('EnvironmentSatisfaction', 4) <= 2:
        score += 2
    if num('JobSatisfaction', 4) <= 2:
        score += 2
    if raw.get('OverTime', '').strip().lower() in ['yes', 'y', 'true', '1']:
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

    max_score = 12
    return min(1.0, score / max_score) * 100

@app.route('/')
def home():
    _, _, _, _, perf = load_all_resources()
    return render_template('home.html', perf=perf)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    model, scaler, encoders, cols, perf = load_all_resources()
    result = None
    note = None

    if request.method == 'POST':
        if not model:
            return "Error: Model Belum Siap!"

        raw_data = request.form.to_dict()
        risk_score = compute_rule_risk(raw_data)
        input_df = pd.DataFrame([raw_data])

        for col, le in encoders.items():
            if col in input_df.columns:
                try:
                    input_df.at[0, col] = safe_label_transform(input_df.at[0, col], le)
                except Exception:
                    input_df.at[0, col] = 0

        input_df = input_df.apply(pd.to_numeric, errors='ignore')
        input_df = input_df.reindex(columns=cols, fill_value=0)

        try:
            input_scaled = scaler.transform(input_df)
            if hasattr(model, 'predict_proba'):
                prob_array = model.predict_proba(input_scaled)[0]
                if hasattr(model, 'classes_') and 1 in model.classes_:
                    idx = list(model.classes_).index(1)
                    confidence = prob_array[idx] if idx < len(prob_array) else 0.0
                else:
                    confidence = prob_array[1] if len(prob_array) > 1 else 0.0
            else:
                confidence = 0.0

            prediction = model.predict(input_scaled)[0]
            status_model = 'Resign' if prediction == 1 else 'Bertahan'
            status_rule = 'Resign' if risk_score >= 55 else 'Bertahan'

            final_status = status_model
            explanation = []
            if status_model != status_rule:
                explanation.append(f"Analisis fitur mendukung {status_rule}, tetapi model prediksi {status_model}.")
            if risk_score >= 70 and status_model == 'Bertahan':
                explanation.append('Skor risiko tinggi (rule-based) walau model memberi Bertahan; periksa intervensi retensi.')
            if status_model == 'Resign' and confidence < 0.4:
                explanation.append('Prediksi Resign dengan confidence rendah, ini indikasi ketidakpastian model.')

            confidence_pct = round(float(confidence) * 100, 2)
            confidence_class = f"w-{min(100, max(0, int(round(confidence_pct))))}"
            result = {
                'status': final_status,
                'confidence': confidence_pct,
                'confidence_class': confidence_class,
                'rule_risk': round(risk_score, 2),
                'model_status': status_model,
                'rule_status': status_rule,
                'explanation': ' '.join(explanation) if explanation else 'Model dan rules konsisten.',
                'performance': perf
            }

            if perf.get('overfit_detected'):
                note = '⚠️ Model menunjukkan tren overfitting, lakukan retraining dengan data lebih segar. (Ditandai oleh sistem)' 

        except Exception as err:
            result = None
            note = f"Error saat prediksi: {err}"

    return render_template('form_prediction.html', result=result, note=note)

if __name__ == '__main__':
    app.run(debug=True)