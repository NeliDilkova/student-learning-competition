from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn

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

    run_name = "random_forest_experiment_v1"

    print(">>> START RANDOM FOREST RUN")
    print("Shapes:", X_train.shape, X_val.shape, y_train.shape, y_val.shape)

    with mlflow.start_run(run_name=run_name):
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )

        print(">>> FITTING MODEL")
        model.fit(X_train, y_train)
        print(">>> FIT DONE")

        print(">>> PREDICTING")
        y_pred = model.predict(X_val)
        print(">>> PRED DONE")

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        print(f"\n[RandomForest] Validation RMSE: {rmse:.5f}")
        print(f"[RandomForest] Validation R2:   {r2:.5f}")

        mlflow.log_metric("val_rmse", float(rmse))
        mlflow.log_metric("val_r2", float(r2))
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("num_numeric_features", len(num_features))
        mlflow.log_param("num_categorical_features", len(cat_features))
        mlflow.log_param("n_estimators", 500)
        mlflow.log_param("max_features", "sqrt")

        importances = model.feature_importances_
        fi_df = pd.DataFrame(
            {"feature": feature_cols, "importance": importances}
        ).sort_values("importance", ascending=False)

        print("\n[RandomForest] Feature Importances:")
        print(fi_df)

        fi_path = project_dir / "models" / "feature_importances_random_forest.csv"
        fi_path.parent.mkdir(parents=True, exist_ok=True)
        fi_df.to_csv(fi_path, index=False)

        mlflow.log_artifact(str(fi_path), artifact_path="feature_importances")
        mlflow.sklearn.log_model(model, artifact_path="model")
