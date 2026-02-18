# src/05_train_models.py

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import xgboost as xgb 
import mlflow
import mlflow.xgboost  # für Autologging


print("Top of file reached")

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    print("__main__ block entered")
    print("__main__ block entered")

    project_dir = get_project_root()
    print("project_dir:", project_dir)
    
    data_features_dir = project_dir / "data" / "features"
    print("data_features_dir:", data_features_dir)

    train_pruned_path = data_features_dir / "train_features_pruned.csv"
    print("train_pruned_path exists?", train_pruned_path.exists())

    test_pruned_path = data_features_dir / "test_features_pruned.csv"
    print("test_pruned_path exists?", test_pruned_path.exists())

    if not train_pruned_path.exists():
        raise FileNotFoundError(
            f"{train_pruned_path} nicht gefunden. Erst 03_features.py und 04_feature_relations.py ausführen."
        )

    train_fe = pd.read_csv(train_pruned_path)

    target_col = "productivity_score"
    id_col = "id"

    if target_col not in train_fe.columns:
        raise KeyError(f"Zielvariable '{target_col}' nicht in TRAIN_FEATURES gefunden")

    # 1) Feature-Listen bestimmen
    # Numerische Features (ohne id, ohne Ziel)
    num_features = train_fe.select_dtypes(
        include=["int64", "float64", "float32"]
    ).columns.tolist()
    num_features = [c for c in num_features if c not in [id_col, target_col]]

    # Kategoriale Features
    cat_features = train_fe.select_dtypes(
        include=["category", "object"]
    ).columns.tolist()
    # Oder explizit:
    # cat_features = ["stress_level_bin", "breaks_per_day_bin", "age_bin"]

    print("\nNumerische Features:")
    print(num_features)
    print("\nKategoriale Features:")
    print(cat_features)

    # 2) Sicherstellen, dass kategoriale Features als category vorliegen
    for col in cat_features:
        train_fe[col] = train_fe[col].astype("category")

    # 3) X, y bauen
    feature_cols = num_features + cat_features
    X = train_fe[feature_cols]
    y = train_fe[target_col]

    # 4) Train/Val-Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5) MLflow-Experiment setzen
    mlflow.set_experiment("student_learning_competition")

    # XGBoost Autologging aktivieren (optional, aber praktisch)
    mlflow.xgboost.autolog()

    with mlflow.start_run():
        # 6) Modell definieren
        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            enable_categorical=True,  # wichtig für category-Features
        )

        # 7) Training mit Early Stopping
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric="rmse",
            verbose=False,
            early_stopping_rounds=50,
        )

        # 8) Evaluation
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        print(f"\nValidation RMSE: {rmse:.5f}")

        # 9) Metriken/Parameter loggen
        mlflow.log_metric("val_rmse", float(rmse))
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("num_numeric_features", len(num_features))
        mlflow.log_param("num_categorical_features", len(cat_features))

        # Modell explizit loggen (optional, Autologging macht es auch)
        mlflow.xgboost.log_model(model, artifact_path="model")
