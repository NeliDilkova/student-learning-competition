# src/06_catboost_tuning.py

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool
import mlflow
import mlflow.catboost

from prepare_to_train import load_train_val_test  # <--- WICHTIG


mlflow.set_tracking_uri("http://127.0.0.1:5000")


if __name__ == "__main__":
    (
        project_dir,
        feature_cols,
        num_features,
        cat_features,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = load_train_val_test()

    mlflow.set_experiment("student_learning_competition")
    run_name = "catboost_random_search_v1"

    # Indizes der kategorialen Features für CatBoost
    cat_idx = [feature_cols.index(c) for c in cat_features] if cat_features else None

    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx)

    base_model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        random_state=42,
        task_type="CPU",
        verbose=False,
    )

    param_grid = {
        "iterations": [600, 800, 1000, 1400],
        "learning_rate": [0.03, 0.02, 0.015, 0.01],
        "depth": [4, 5, 6, 7, 8],
        "l2_leaf_reg": [1, 3, 5, 7, 9],
        "random_strength": [0.5, 1.0, 2.0, 5.0],
        "bagging_temperature": [0.0, 0.5, 1.0],
        "border_count": [128, 254],
    }

    with mlflow.start_run(run_name=run_name):
        search_results = base_model.randomized_search(
            param_grid,
            X=train_pool,
            y=None,
            cv=3,
            n_iter=25,
            partition_random_seed=42,
            verbose=True,
            calc_cv_statistics=True,
            refit=True,  # nach CV auf train_pool refitten
        )

        best_params = search_results["params"]
        print("\nBeste gefundene Hyperparameter:")
        print(best_params)

        # Validation-Metriken mit dem refit-Modell (train_pool)
        y_pred_val = base_model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        val_r2 = r2_score(y_val, y_pred_val)

        print(f"\n[Tuned CatBoost] Validation RMSE: {val_rmse:.5f}")
        print(f"[Tuned CatBoost] Validation R2:   {val_r2:.5f}")

        # Jetzt finales Modell mit besten Parametern auf TRAIN+VAL neu trainieren
        X_train_full = pd.concat([X_train, X_val], axis=0)
        y_train_full = pd.concat([y_train, y_val], axis=0)

        train_full_pool = Pool(
            X_train_full,
            y_train_full,
            cat_features=cat_idx,
        )

        final_model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            random_state=42,
            task_type="CPU",
            verbose=False,
            **best_params,
        )

        final_model.fit(
            train_full_pool,
            verbose=False,
        )

        # Test-Metriken
        y_pred_test = final_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)

        print(f"\n[Tuned CatBoost] Test RMSE: {test_rmse:.5f}")
        print(f"[Tuned CatBoost] Test R2:   {test_r2:.5f}")

        # MLflow-Logging
        mlflow.log_metric("val_rmse", float(val_rmse))
        mlflow.log_metric("val_r2", float(val_r2))
        mlflow.log_metric("test_rmse", float(test_rmse))
        mlflow.log_metric("test_r2", float(test_r2))

        for k, v in best_params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("num_numeric_features", len(num_features))
        mlflow.log_param("num_categorical_features", len(cat_features))

        # Feature Importances auf TRAIN+VAL
        fi_importances = final_model.get_feature_importance(train_full_pool)
        fi_df = pd.DataFrame(
            {"feature": feature_cols, "importance": fi_importances}
        ).sort_values("importance", ascending=False)

        print("\n[Tuned CatBoost] Feature Importances (Train+Val):")
        print(fi_df)

        fi_path = project_dir / "models" / "feature_importances_catboost_tuned.csv"
        fi_path.parent.mkdir(parents=True, exist_ok=True)
        fi_df.to_csv(fi_path, index=False)

        mlflow.log_artifact(str(fi_path), artifact_path="feature_importances")
        mlflow.catboost.log_model(final_model, artifact_path="model_tuned")
