# src/06_train_xgboost.py

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import mlflow
import mlflow.xgboost

from prepare_to_train import load_train_data  # gemeinsamer Loader

mlflow.set_tracking_uri("http://127.0.0.1:5000")

if __name__ == "__main__":
    (
        project_dir,
        feature_cols,
        num_features,
        cat_features,
        X_train,
        X_val,
        y_train,
        y_val,
    ) = load_train_data()

    mlflow.set_experiment("student_learning_competition")
#    mlflow.xgboost.autolog()

    run_name = "xgboost_experiment_v1"

    with mlflow.start_run(run_name=run_name):
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

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        print(f"\n[XGBoost] Validation RMSE: {rmse:.5f}")
        print(f"[XGBoost] Validation R2:   {r2:.5f}")

        mlflow.log_metric("val_rmse", float(rmse))
        mlflow.log_metric("val_r2", float(r2))
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("num_numeric_features", len(num_features))
        mlflow.log_param("num_categorical_features", len(cat_features))

        importances = model.feature_importances_
        fi_df = pd.DataFrame(
            {"feature": feature_cols, "importance": importances}
        ).sort_values("importance", ascending=False)

        print("\n[XGBoost] Feature Importances:")
        print(fi_df)

        fi_path = project_dir / "models" / "feature_importances_xgb.csv"
        fi_path.parent.mkdir(parents=True, exist_ok=True)
        fi_df.to_csv(fi_path, index=False)

        mlflow.log_artifact(str(fi_path), artifact_path="feature_importances")
        mlflow.xgboost.log_model(model, name="model")
