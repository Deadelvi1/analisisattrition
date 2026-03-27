import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import mlflow
import os


def evaluate_sets(model, X, y, dataset_name):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    metrics = {
        'accuracy': accuracy_score(y, pred),
        'precision': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0)
    }
    if prob is not None and len(np.unique(y)) > 1:
        try:
            metrics['roc_auc'] = roc_auc_score(y, prob)
        except Exception:
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = int(tn), int(fp), int(fn), int(tp)
    return metrics


def train_model():
    if not os.path.exists('model'):
        os.makedirs('model')

    df = pd.read_csv('employee_data.csv')

    drop_cols = ['EmployeeId', 'Predicted_Attrition', 'Resign_Probability', 'Risk_Level', 'EmployeeCount', 'Over18', 'StandardHours']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    categorical_cols = df.select_dtypes(include=['object']).columns
    encoders = {}
    for col in categorical_cols:
        if col != 'Attrition':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    X = df.drop('Attrition', axis=1)
    y = df['Attrition'].apply(lambda x: 1 if str(x).strip().lower() in ['1', '1.0', 'yes', 'true'] else 0)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.17647059, random_state=42, stratify=y_temp)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    mlflow.set_experiment('HR_Attrition_Professional')

    best_params = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_leaf': 5,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }

    with mlflow.start_run():
        model = RandomForestClassifier(**best_params)
        model.fit(X_train_scaled, y_train)

        train_metrics = evaluate_sets(model, X_train_scaled, y_train, 'train')
        val_metrics = evaluate_sets(model, X_val_scaled, y_val, 'val')
        test_metrics = evaluate_sets(model, X_test_scaled, y_test, 'test')

        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1', n_jobs=-1)

        overfit_gap = train_metrics['accuracy'] - test_metrics['accuracy']
        overfit_flag = overfit_gap > 0.12

        if overfit_flag:
            best_params.update({'max_depth': 7, 'min_samples_leaf': 10, 'n_estimators': 150})
            model = RandomForestClassifier(**best_params)
            model.fit(X_train_scaled, y_train)
            train_metrics = evaluate_sets(model, X_train_scaled, y_train, 'train')
            val_metrics = evaluate_sets(model, X_val_scaled, y_val, 'val')
            test_metrics = evaluate_sets(model, X_test_scaled, y_test, 'test')
            mlflow.set_tag('overfitting_mitigation', 'applied')

        mlflow.log_params(best_params)
        mlflow.log_metric('train_accuracy', train_metrics['accuracy'])
        mlflow.log_metric('val_accuracy', val_metrics['accuracy'])
        mlflow.log_metric('test_accuracy', test_metrics['accuracy'])
        mlflow.log_metric('train_f1', train_metrics['f1'])
        mlflow.log_metric('val_f1', val_metrics['f1'])
        mlflow.log_metric('test_f1', test_metrics['f1'])
        mlflow.log_metric('mean_cv_f1', float(np.mean(cv_scores)))
        mlflow.log_metric('std_cv_f1', float(np.std(cv_scores)))
        mlflow.log_metric('overfit_gap', overfit_gap)
        mlflow.log_metric('overfit_detected', int(overfit_flag))

        mlflow.sklearn.log_model(model, 'random_forest_model')

        joblib.dump(model, 'model/attrition_model.pkl')
        joblib.dump(scaler, 'model/scaler.pkl')
        joblib.dump(encoders, 'model/encoders.pkl')
        joblib.dump(X.columns.tolist(), 'model/model_columns.pkl')
        joblib.dump({
            'train': {k: round(v * 100, 2) for k, v in train_metrics.items() if isinstance(v, (int, float)) and v is not None and k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']},
            'val': {k: round(v * 100, 2) for k, v in val_metrics.items() if isinstance(v, (int, float)) and v is not None and k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']},
            'test': {k: round(v * 100, 2) for k, v in test_metrics.items() if isinstance(v, (int, float)) and v is not None and k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']},
            'overfit_gap': round(overfit_gap * 100, 2),
            'overfit_detected': bool(overfit_flag),
            'cv_f1_mean': round(float(np.mean(cv_scores)) * 100, 2),
            'cv_f1_std': round(float(np.std(cv_scores)) * 100, 2)
        }, 'model/performance.pkl')

    print('✅ Model Berhasil Dilatih & Dicatat di MLflow!')
    print('  - Train Accuracy:', round(train_metrics['accuracy'], 4))
    print('  - Validation Accuracy:', round(val_metrics['accuracy'], 4))
    print('  - Test Accuracy:', round(test_metrics['accuracy'], 4))
    print('  - Overfit Gap:', round(overfit_gap, 4), 'Overfit:', overfit_flag)


if __name__ == '__main__':
    train_model()