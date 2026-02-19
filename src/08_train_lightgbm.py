from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import mlflow
import mlflow.sklearn

from prepare_to_train import load_train_data


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

    run_name = "lightgbm_experiment_v1"

    with mlflow.start_run(run_name=run_name):
        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            objective="regression",
        )

        # Falls nötig: kategorische Spalten in category casten
        # X_train[cat_features] = X_train[cat_features].astype("category")
        # X_val[cat_features] = X_val[cat_features].astype("category")

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric="rmse",
        )

        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        print(f"\n[LightGBM] Validation RMSE: {rmse:.5f}")
        print(f"[LightGBM] Validation R2:   {r2:.5f}")

        mlflow.log_metric("val_rmse", float(rmse))
        mlflow.log_metric("val_r2", float(r2))
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("num_numeric_features", len(num_features))
        mlflow.log_param("num_categorical_features", len(cat_features))

        importances = model.feature_importances_
        fi_df = pd.DataFrame(
            {"feature": feature_cols, "importance": importances}
        ).sort_values("importance", ascending=False)

        print("\n[LightGBM] Feature Importances:")
        print(fi_df)

        fi_path = project_dir / "models" / "feature_importances_lgbm.csv"
        fi_path.parent.mkdir(parents=True, exist_ok=True)
        fi_df.to_csv(fi_path, index=False)

        mlflow.log_artifact(str(fi_path), artifact_path="feature_importances")
        mlflow.sklearn.log_model(model, artifact_path="model")
