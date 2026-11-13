import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from data_preprocessing import prepare_train_test_data


def evaluate_predictions(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}


def train_and_evaluate():
    print("1. Veri yükleniyor ve ön işleme hazırlanıyor...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_data()
    print(f"Eğitim seti boyutu: {X_train.shape}, Test seti boyutu: {X_test.shape}")

    models = {
        "Ridge_Regression": Ridge(alpha=1.0),
        "Random_Forest": RandomForestRegressor(
            n_estimators=150, max_depth=15, random_state=42, n_jobs=-1
        ),
        "XGBoost_Base": XGBRegressor(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
    }

    results = {}
    best_model_name = None
    best_r2 = -np.inf
    best_pipeline = None

    print("\n2. Modeller eğitiliyor ve değerlendiriliyor...")
    for name, regressor in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", regressor)
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = evaluate_predictions(y_test, preds)
        results[name] = metrics
        print(f" -> {name}: RMSE={metrics['RMSE']}, MAE={metrics['MAE']}, R2={metrics['R2']}")

        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            best_model_name = name
            best_pipeline = pipe

    print("\n3. XGBoost için Hiperparametre Optimizasyonu (RandomizedSearchCV) çalıştırılıyor...")
    xgb_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(random_state=42, n_jobs=-1))
    ])

    param_distributions = {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [4, 6, 8],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
    }

    search = RandomizedSearchCV(
        xgb_pipe,
        param_distributions=param_distributions,
        n_iter=8,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)
    tuned_preds = search.best_estimator_.predict(X_test)
    tuned_metrics = evaluate_predictions(y_test, tuned_preds)
    results["XGBoost_Tuned"] = tuned_metrics
    print(f" -> XGBoost_Tuned: RMSE={tuned_metrics['RMSE']}, MAE={tuned_metrics['MAE']}, R2={tuned_metrics['R2']}")

    if tuned_metrics["R2"] >= best_r2:
        best_model_name = "XGBoost_Tuned"
        best_pipeline = search.best_estimator_

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    model_path = "models/best_los_model.joblib"
    joblib.dump(best_pipeline, model_path)

    with open("reports/metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nEn iyi model ({best_model_name}) '{model_path}' konumuna kaydedildi!")
    print("Metrikler 'reports/metrics.json' dosyasına yazıldı.")


if __name__ == "__main__":
    train_and_evaluate()
