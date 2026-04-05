import os
import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn
import joblib
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


def train():
    print("🚀 MODEL TRAINING STARTED")

    os.environ["MLFLOW_TRACKING_USERNAME"] = "deadelvina9"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "a63253428df5fd08b4778e5e4ef9f7420da87ab9"

    dagshub.init(
        repo_owner="deadelvina9",
        repo_name="attrition-mlops",
        mlflow=True
    )

    df = pd.read_csv("employee_data.csv")

    df["Attrition"] = pd.to_numeric(df["Attrition"], errors="coerce")

    df_train = df[df["Attrition"].notna()].copy()
    df_unlabeled = df[df["Attrition"].isna()].copy()

    df_train["Attrition"] = df_train["Attrition"].astype(int)

    drop_cols = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
    df_train = df_train.drop(columns=drop_cols, errors="ignore")
    df_unlabeled = df_unlabeled.drop(columns=drop_cols, errors="ignore")

    X = df_train.drop("Attrition", axis=1)
    y = df_train["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    model = RandomForestClassifier(random_state=42, class_weight="balanced")

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [5, 10, None]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1", n_jobs=-1)

    with mlflow.start_run():
        print("⚙️ TRAINING MODEL...")
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print("📊 EVALUATION:")
        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1 Score  : {f1:.4f}")

        mlflow.log_param("model", "RandomForest")
        mlflow.log_params(grid.best_params_)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        print("📦 LOGGING MODEL TO MLFLOW...")
        mlflow.sklearn.log_model(
            best_model,
            name="model",
            registered_model_name="rf_model_tuning"
        )

        os.makedirs("model", exist_ok=True)
        joblib.dump(best_model, "model/rf_model_tuning_latest.pkl")

        # Calculate additional metrics
        y_train_pred = best_model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        overfit_gap = train_acc - acc

        cv_scores = cross_val_score(best_model, X, y, cv=3, scoring='f1')
        cv_f1_mean = cv_scores.mean()
        cv_f1_std = cv_scores.std()

        performance = {
            "test": {
                "accuracy": round(acc * 100, 2),
                "f1": round(f1 * 100, 2),
                "precision": round(prec * 100, 2),
                "recall": round(rec * 100, 2)
            },
            "val": {"accuracy": round(train_acc * 100, 2)},
            "overfit_gap": round(overfit_gap * 100, 2),
            "cv_f1_mean": round(cv_f1_mean * 100, 1),
            "cv_f1_std": round(cv_f1_std * 100, 1)
        }

        with open("model/performance.json", "w") as f:
            json.dump(performance, f)

        print("📊 PERFORMANCE METRICS SAVED:")
        print(f"Test Accuracy: {performance['test']['accuracy']}%")
        print(f"Test F1: {performance['test']['f1']}%")
        print(f"Test Precision: {performance['test']['precision']}%")
        print(f"Test Recall: {performance['test']['recall']}%")
        print(f"Val Accuracy: {performance['val']['accuracy']}%")
        print(f"Overfit Gap: {performance['overfit_gap']}%")
        print(f"CV F1 Mean: {performance['cv_f1_mean']}% ± {performance['cv_f1_std']}%")

    print("✅ MODEL TRAINING FINISHED")


def get_model():
    print("📥 LOADING MODEL FROM MLFLOW")

    uri = "https://dagshub.com/deadelvina9/attrition-mlops.mlflow"
    mlflow.set_tracking_uri(uri)

    model_uri = "models:/rf_model_tuning/2"
    model = mlflow.sklearn.load_model(model_uri)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/rf_model_latest.pkl")

    return model


def load_model():
    print("📦 LOADING LOCAL MODEL")
    return joblib.load("model/rf_model_latest.pkl")


if __name__ == "__main__":
    train()