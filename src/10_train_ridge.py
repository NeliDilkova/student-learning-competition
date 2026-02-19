from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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

    print(">>> NUM FEATURES:", num_features)

    X_train_num = X_train[num_features].copy()
    X_val_num = X_val[num_features].copy()

    print(">>> SHAPES:", X_train_num.shape, X_val_num.shape)
    print(">>> NaNs in X_train_num:", X_train_num.isna().sum().sum())
    print(">>> NaNs in y_train:", np.isnan(y_train).sum())

    mlflow.set_experiment("student_learning_competition")

    print(">>> START ELASTICNET RUN")

    # Pipeline: Imputer -> StandardScaler -> ElasticNet
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "enet",
                ElasticNet(
                    alpha=0.01,      # deutlich stärkere Regularisierung
                    l1_ratio=0.2,    # überwiegend L2
                    max_iter=1000,   # klein halten
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

    print(f"\n[ElasticNet] Validation RMSE: {rmse:.5f}")
    print(f"[ElasticNet] Validation R2:   {r2:.5f}")

    mlflow.log_metric("val_rmse", float(rmse))
    mlflow.log_metric("val_r2", float(r2))
    mlflow.log_param("num_features", len(num_features))
    mlflow.log_param("model_type", "ElasticNet")
    mlflow.log_param("alpha", 0.0005)
    mlflow.log_param("l1_ratio", 0.5)

    enet = model.named_steps["enet"]
    coefs = enet.coef_
    fi_df = pd.DataFrame(
        {"feature": num_features, "importance": coefs}
    ).sort_values("importance", key=lambda s: np.abs(s), ascending=False)

    print("\n[ElasticNet] Coefficients (Importances):")
    print(fi_df)

    fi_path = project_dir / "models" / "feature_importances_elasticnet.csv"
    fi_path.parent.mkdir(parents=True, exist_ok=True)
    fi_df.to_csv(fi_path, index=False)

    mlflow.log_artifact(str(fi_path), artifact_path="feature_importances")
    mlflow.sklearn.log_model(model, artifact_path="model")
