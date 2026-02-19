# src/05_train_models.py

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import mlflow
import mlflow.xgboost  # für Autologging


print("Top of file reached")


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


if __name__ == "__main__":
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
    num_features = (
        train_fe.select_dtypes(include=["int64", "float64", "float32"])
        .columns.tolist()
    )
    num_features = [c for c in num_features if c not in [id_col, target_col]]

    cat_features = (
        train_fe.select_dtypes(include=["category", "object"])
        .columns.tolist()
    )

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

    mlflow.xgboost.autolog()

    run_name = "xgboost_baseline_v1"

    with mlflow.start_run(run_name=run_name):
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
            enable_categorical=True,
            eval_metric="rmse",
        )

        # 7) Training (ohne early_stopping_rounds, da deine Version das nicht im fit() erwartet)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        # 8) Evaluation: RMSE und R²
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        print(f"\nValidation RMSE: {rmse:.5f}")
        print(f"Validation R2:   {r2:.5f}")

        mlflow.log_metric("val_rmse", float(rmse))
        mlflow.log_metric("val_r2", float(r2))  # R² loggen

        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("num_numeric_features", len(num_features))
        mlflow.log_param("num_categorical_features", len(cat_features))

        # 9) Feature Importances (Beitrage der einzelnen Variablen)
        importances = model.feature_importances_  # array in Feature-Reihenfolge[web:439][web:432]
        fi_df = pd.DataFrame(
            {"feature": feature_cols, "importance": importances}
        ).sort_values("importance", ascending=False)

        print("\nFeature Importances:")
        print(fi_df)

        # als CSV speichern und als Artifact loggen
        fi_path = project_dir / "models" / "feature_importances_xgb.csv"
        fi_path.parent.mkdir(parents=True, exist_ok=True)
        fi_df.to_csv(fi_path, index=False)

        mlflow.log_artifact(str(fi_path), artifact_path="feature_importances")  #[web:437][web:441]

        # Modell explizit loggen (Autologging macht es eigentlich schon)
        mlflow.xgboost.log_model(model, artifact_path="model")
