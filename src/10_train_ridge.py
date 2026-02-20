# src/10_train_elasticnet.py

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

ALPHA    = 0.01
L1_RATIO = 0.2
MAX_ITER = 5000
TOL      = 1e-4

def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)

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

    X_train_num = clean_numeric(X_train[num_features].copy())
    X_val_num   = clean_numeric(X_val[num_features].copy())

    y_train_arr = np.array(y_train, dtype=np.float64)
    y_val_arr   = np.array(y_val,   dtype=np.float64)

    mlflow.set_experiment("student_learning_competition")

    with mlflow.start_run(run_name="elasticnet_experiment_v1"):

        model = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("enet", ElasticNet(
                alpha=ALPHA,
                l1_ratio=L1_RATIO,
                max_iter=MAX_ITER,
                tol=TOL,
                selection="random",
                random_state=42,
            )),
        ])

        print(">>> FITTING MODEL ...")
        model.fit(X_train_num, y_train_arr)
        print(">>> FIT DONE")

        y_pred = model.predict(X_val_num)
        rmse   = np.sqrt(mean_squared_error(y_val_arr, y_pred))
        r2     = r2_score(y_val_arr, y_pred)

        print(f"\n[ElasticNet] RMSE: {rmse:.5f}")
        print(f"[ElasticNet] R²:   {r2:.5f}")

        enet = model.named_steps["enet"]
        print(f"[ElasticNet] Konvergiert nach {enet.n_iter_} Iterationen")

        mlflow.log_param("model_type",   "ElasticNet")
        mlflow.log_param("alpha",        ALPHA)
        mlflow.log_param("l1_ratio",     L1_RATIO)
        mlflow.log_param("max_iter",     MAX_ITER)
        mlflow.log_param("tol",          TOL)
        mlflow.log_param("selection",    "random")
        mlflow.log_param("num_features", len(num_features))
        mlflow.log_param("n_iter",       int(enet.n_iter_))

        mlflow.log_metric("val_rmse",     float(rmse))
        mlflow.log_metric("val_r2",       float(r2))
        mlflow.log_metric("n_zero_coefs", int((enet.coef_ == 0).sum()))

        fi_df = (
            pd.DataFrame({"feature": num_features, "coefficient": enet.coef_})
            .assign(abs_coef=lambda df: df["coefficient"].abs())
            .sort_values("abs_coef", ascending=False)
            .drop(columns="abs_coef")
        )
        print("\n[ElasticNet] Koeffizienten:")
        print(fi_df.to_string(index=False))

        fi_path = project_dir / "models" / "feature_importances_elasticnet.csv"
        fi_path.parent.mkdir(parents=True, exist_ok=True)
        fi_df.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path), artifact_path="feature_importances")

        mlflow.sklearn.log_model(model, artifact_path="model")

        print("\n>>> MLflow Run abgeschlossen ✓")
