from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

    X_train_num = X_train[num_features].copy()
    X_val_num = X_val[num_features].copy()

    mlflow.set_experiment("student_learning_competition")

    print(">>> START SGD-ELASTICNET RUN")
    print("Shapes:", X_train_num.shape, X_val_num.shape, y_train.shape, y_val.shape)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "sgd_elastic",
                SGDRegressor(
                    penalty="elasticnet",
                    alpha=0.0001,      # Stärke der Regularisierung
                    l1_ratio=0.5,      # Anteil L1 vs. L2
                    max_iter=1000,
                    tol=1e-3,
                    random_state=42,
                ),
            ),
        ]
    )

    print(">>> FITTING MODEL")
    model.fit(X_train_num, y_train)
    print(">>> FIT DONE")

    print(">>> PREDICTING")
    y_pred = model.predict(X_val_num)
    print(">>> PRED DONE")

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    print(f"\n[SGD-ElasticNet] Validation RMSE: {rmse:.5f}")
    print(f"[SGD-ElasticNet] Validation R2:   {r2:.5f}")

    mlflow.log_metric("val_rmse", float(rmse))
    mlflow.log_metric("val_r2", float(r2))
    mlflow.log_param("num_features", len(num_features))
    mlflow.log_param("model_type", "SGD_ElasticNet")
    mlflow.log_param("alpha", 0.0001)
    mlflow.log_param("l1_ratio", 0.5)

    sgd = model.named_steps["sgd_elastic"]
    coefs = sgd.coef_
    fi_df = pd.DataFrame(
        {"feature": num_features, "importance": coefs}
    ).sort_values("importance", key=lambda s: np.abs(s), ascending=False)

    print("\n[SGD-ElasticNet] Coefficients (Importances):")
    print(fi_df)

    fi_path = project_dir / "models" / "feature_importances_sgd_elasticnet.csv"
    fi_path.parent.mkdir(parents=True, exist_ok=True)
    fi_df.to_csv(fi_path, index=False)

    mlflow.log_artifact(str(fi_path), artifact_path="feature_importances")
    mlflow.sklearn.log_model(model, artifact_path="model")
