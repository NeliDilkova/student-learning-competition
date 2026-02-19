# src/12_evaluate_on_test.py

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.catboost

from prepare_to_train import load_train_val_test

mlflow.set_tracking_uri("http://127.0.0.1:5000")
EXPERIMENT_NAME = "student_learning_competition"


def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2


def eval_xgboost(project_dir, feature_cols, num_features, cat_features,
                 X_train, X_val, X_test, y_train, y_val, y_test):
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = "xgboost_eval_train_val_test"

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
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        val_rmse, val_r2 = eval_metrics(y_val, y_val_pred)
        test_rmse, test_r2 = eval_metrics(y_test, y_test_pred)

        print(f"[XGB] Val RMSE:  {val_rmse:.5f}, Val R2:  {val_r2:.5f}")
        print(f"[XGB] Test RMSE: {test_rmse:.5f}, Test R2: {test_r2:.5f}")

        mlflow.log_metric("val_rmse", float(val_rmse))
        mlflow.log_metric("val_r2", float(val_r2))
        mlflow.log_metric("test_rmse", float(test_rmse))
        mlflow.log_metric("test_r2", float(test_r2))
        mlflow.log_param("model", "xgboost_eval")
        mlflow.log_param("num_features", len(feature_cols))

        mlflow.xgboost.log_model(model, artifact_path="model")


def eval_lightgbm(project_dir, feature_cols, num_features, cat_features,
                  X_train, X_val, X_test, y_train, y_val, y_test):
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = "lightgbm_eval_train_val_test"

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

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
        )

        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        val_rmse, val_r2 = eval_metrics(y_val, y_val_pred)
        test_rmse, test_r2 = eval_metrics(y_test, y_test_pred)

        print(f"[LGBM] Val RMSE:  {val_rmse:.5f}, Val R2:  {val_r2:.5f}")
        print(f"[LGBM] Test RMSE: {test_rmse:.5f}, Test R2: {test_r2:.5f}")

        mlflow.log_metric("val_rmse", float(val_rmse))
        mlflow.log_metric("val_r2", float(val_r2))
        mlflow.log_metric("test_rmse", float(test_rmse))
        mlflow.log_metric("test_r2", float(test_r2))
        mlflow.log_param("model", "lightgbm_eval")
        mlflow.log_param("num_features", len(feature_cols))

        mlflow.sklearn.log_model(model, artifact_path="model")


def eval_catboost(project_dir, feature_cols, num_features, cat_features,
                  X_train, X_val, X_test, y_train, y_val, y_test):
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = "catboost_eval_train_val_test"

    cat_idx = [feature_cols.index(c) for c in cat_features]

    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx)
    test_pool = Pool(X_test, y_test, cat_features=cat_idx)

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

        y_val_pred = model.predict(val_pool)
        y_test_pred = model.predict(test_pool)

        val_rmse, val_r2 = eval_metrics(y_val, y_val_pred)
        test_rmse, test_r2 = eval_metrics(y_test, y_test_pred)

        print(f"[CatBoost] Val RMSE:  {val_rmse:.5f}, Val R2:  {val_r2:.5f}")
        print(f"[CatBoost] Test RMSE: {test_rmse:.5f}, Test R2: {test_r2:.5f}")

        mlflow.log_metric("val_rmse", float(val_rmse))
        mlflow.log_metric("val_r2", float(val_r2))
        mlflow.log_metric("test_rmse", float(test_rmse))
        mlflow.log_metric("test_r2", float(test_r2))
        mlflow.log_param("model", "catboost_eval")
        mlflow.log_param("num_features", len(feature_cols))

        mlflow.catboost.log_model(model, artifact_path="model")


def eval_random_forest(project_dir, feature_cols, num_features, cat_features,
                       X_train, X_val, X_test, y_train, y_val, y_test):
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = "random_forest_eval_train_val_test"

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

        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        val_rmse, val_r2 = eval_metrics(y_val, y_val_pred)
        test_rmse, test_r2 = eval_metrics(y_test, y_test_pred)

        print(f"[RF] Val RMSE:  {val_rmse:.5f}, Val R2:  {val_r2:.5f}")
        print(f"[RF] Test RMSE: {test_rmse:.5f}, Test R2: {test_r2:.5f}")

        mlflow.log_metric("val_rmse", float(val_rmse))
        mlflow.log_metric("val_r2", float(val_r2))
        mlflow.log_metric("test_rmse", float(test_rmse))
        mlflow.log_metric("test_r2", float(test_r2))
        mlflow.log_param("model", "random_forest_eval")
        mlflow.log_param("num_features", len(feature_cols))

        mlflow.sklearn.log_model(model, artifact_path="model")


def eval_benchmark_xgboost(project_dir, feature_cols, num_features, cat_features,
                           X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Benchmark-XGBoost mit Kaggle-ähnlichen Parametern auf Train/Val/Test,
    aber mit deinem Feature-Set.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = "benchmark_xgboost_eval_train_val_test"

    with mlflow.start_run(run_name=run_name):
        model = XGBRegressor(
            n_estimators=1500,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
            objective="reg:squarederror",
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        val_rmse, val_r2 = eval_metrics(y_val, y_val_pred)
        test_rmse, test_r2 = eval_metrics(y_test, y_test_pred)

        print(f"[Benchmark XGB] Val RMSE:  {val_rmse:.5f}, Val R2:  {val_r2:.5f}")
        print(f"[Benchmark XGB] Test RMSE: {test_rmse:.5f}, Test R2: {test_r2:.5f}")

        mlflow.log_metric("val_rmse", float(val_rmse))
        mlflow.log_metric("val_r2", float(val_r2))
        mlflow.log_metric("test_rmse", float(test_rmse))
        mlflow.log_metric("test_r2", float(test_r2))
        mlflow.log_param("model", "benchmark_xgboost_eval")
        mlflow.log_param("num_features", len(feature_cols))

        mlflow.sklearn.log_model(model, artifact_path="model")


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
    ) = load_train_val_test(val_size=0.2, test_size=0.2, random_state=42)

    eval_xgboost(project_dir, feature_cols, num_features, cat_features,
                 X_train, X_val, X_test, y_train, y_val, y_test)

    eval_lightgbm(project_dir, feature_cols, num_features, cat_features,
                  X_train, X_val, X_test, y_train, y_val, y_test)

    eval_catboost(project_dir, feature_cols, num_features, cat_features,
                  X_train, X_val, X_test, y_train, y_val, y_test)

    eval_random_forest(project_dir, feature_cols, num_features, cat_features,
                       X_train, X_val, X_test, y_train, y_val, y_test)

    eval_benchmark_xgboost(project_dir, feature_cols, num_features, cat_features,
                           X_train, X_val, X_test, y_train, y_val, y_test)
