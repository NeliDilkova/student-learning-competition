from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool
import mlflow
import mlflow.catboost

from prepare_to_train import load_train_data

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

    run_name = "catboost_experiment_v1"

    # Indizes der kategorialen Features für CatBoost
    cat_idx = [feature_cols.index(c) for c in cat_features]

    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx)

    with mlflow.start_run(run_name=run_name):
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_state=42,
            task_type="CPU",
            verbose=False,
        )

        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            verbose=False,
        )

        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        print(f"\n[CatBoost] Validation RMSE: {rmse:.5f}")
        print(f"[CatBoost] Validation R2:   {r2:.5f}")

        mlflow.log_metric("val_rmse", float(rmse))
        mlflow.log_metric("val_r2", float(r2))
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("num_numeric_features", len(num_features))
        mlflow.log_param("num_categorical_features", len(cat_features))

        importances = model.get_feature_importance(train_pool)
        fi_df = pd.DataFrame(
            {"feature": feature_cols, "importance": importances}
        ).sort_values("importance", ascending=False)

        print("\n[CatBoost] Feature Importances:")
        print(fi_df)

        fi_path = project_dir / "models" / "feature_importances_catboost.csv"
        fi_path.parent.mkdir(parents=True, exist_ok=True)
        fi_df.to_csv(fi_path, index=False)

        mlflow.log_artifact(str(fi_path), artifact_path="feature_importances")
        mlflow.catboost.log_model(model, artifact_path="model")
