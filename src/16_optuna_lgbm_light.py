# src/16_optuna_lgbm_light.py
"""
Optuna-Tuning für LightGBM — CPU-schonend für Python 3.9.
Ziel: Unter Leaderboard-Platz 1 (RMSE 0.03691).
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import mlflow

# ── Konstanten ──────────────────────────────────────────────────────────────────
N_TRIALS     = 50          # 50 reichen für gute Ergebnisse
N_SPLITS     = 5
RANDOM_STATE = 42
MLFLOW_URI   = "http://127.0.0.1:5000"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("student_learning_competition")


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_data(project_dir: Path):
    feat_dir = project_dir / "data" / "features"
    train = pd.read_csv(feat_dir / "train_features.csv")
    test  = pd.read_csv(feat_dir / "test_features.csv")

    target = "productivity_score"
    id_col = "id"

    test_ids = test[id_col].values if id_col in test.columns else np.arange(len(test))

    # Kategoriale Features: nur gender (wie Benchmark)
    cat_cols = ["gender"] if "gender" in train.columns else []

    drop_cols = {target, id_col}
    feature_cols = [c for c in train.columns if c not in drop_cols]

    X = train[feature_cols].copy()
    y = train[target].values
    X_test = test[feature_cols].copy() if id_col in test.columns else test.copy()

    # gender als category
    for col in cat_cols:
        if col in X.columns:
            X[col]      = X[col].astype("category")
            X_test[col] = X_test[col].astype("category")

    return X, y, X_test, test_ids, feature_cols, cat_cols


def objective(trial, X, y, cat_cols, n_splits=N_SPLITS):
    params = {
        "objective"        : "regression",
        "metric"           : "rmse",
        "verbosity"        : -1,
        "n_jobs"           : 2,          # max 2 Kerne → CPU nicht überlasten
        "random_state"     : RANDOM_STATE,
        "n_estimators"     : 3000,
        # Suchraum
        "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
        "max_depth"        : trial.suggest_int("max_depth", 3, 7),
        "num_leaves"       : trial.suggest_int("num_leaves", 4, 50),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq"   : trial.suggest_int("subsample_freq", 1, 10),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for train_idx, val_idx in kf.split(X):
        X_tr = X.iloc[train_idx].copy()
        X_v  = X.iloc[val_idx].copy()
        y_tr = y[train_idx]
        y_v  = y[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            categorical_feature = cat_cols if cat_cols else "auto",
            eval_set            = [(X_v, y_v)],
            callbacks           = [
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        scores.append(rmse(y_v, model.predict(X_v)))

    return float(np.mean(scores))


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[1]
    print("Lade Daten ...")
    X, y, X_test, test_ids, feature_cols, cat_cols = load_data(project_dir)
    print(f"  X: {X.shape}  |  X_test: {X_test.shape}")
    print(f"  Kategoriale Features: {cat_cols}")

    # ── Optuna-Studie ────────────────────────────────────────────────────────────
    print(f"\nStarte Optuna-Tuning ({N_TRIALS} Trials, sequenziell) ...")
    study = optuna.create_study(
        direction = "minimize",
        sampler   = optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective(trial, X, y, cat_cols),
        n_trials          = N_TRIALS,
        n_jobs            = 1,          # sequenziell → kein Crash
        show_progress_bar = True,
    )

    best_params  = study.best_params
    best_cv_rmse = study.best_value
    print(f"\nBeste Parameter : {best_params}")
    print(f"Bester CV-RMSE  : {best_cv_rmse:.5f}")

    # ── Finales Modell: KFold auf allen Trainingsdaten ───────────────────────────
    print("\nTrainiere finales Modell (KFold OOF + Test-Predictions) ...")
    final_params = {
        "objective"        : "regression",
        "metric"           : "rmse",
        "verbosity"        : -1,
        "n_jobs"           : 2,
        "random_state"     : RANDOM_STATE,
        "n_estimators"     : 3000,
        **best_params,
    }

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    n_samples      = len(X)
    oof_preds      = np.zeros(n_samples)
    test_preds_all = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"  Fold {fold+1}/{N_SPLITS} ...", end=" ", flush=True)
        X_tr = X.iloc[train_idx].copy()
        X_v  = X.iloc[val_idx].copy()
        y_tr = y[train_idx]
        y_v  = y[val_idx]

        model = lgb.LGBMRegressor(**final_params)
        model.fit(
            X_tr, y_tr,
            categorical_feature = cat_cols if cat_cols else "auto",
            eval_set            = [(X_v, y_v)],
            callbacks           = [
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        oof_preds[val_idx] = model.predict(X_v)
        test_preds_all.append(model.predict(X_test))
        print(f"RMSE = {rmse(y_v, model.predict(X_v)):.5f}")

    oof_rmse   = rmse(y, oof_preds)
    test_preds = np.mean(test_preds_all, axis=0)
    print(f"\nOOF-RMSE (gesamt): {oof_rmse:.5f}")

    # ── Submission speichern ─────────────────────────────────────────────────────
    submission = pd.DataFrame({
        "id"               : test_ids,
        "productivity_score": test_preds,
    })
    out_path = project_dir / "models" / "submission_lgbm_optuna.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"Submission gespeichert: {out_path}")

    # ── MLflow-Logging ───────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="lgbm_optuna_v1"):
        mlflow.log_metric("optuna_cv_rmse", best_cv_rmse)
        mlflow.log_metric("oof_rmse",       oof_rmse)
        mlflow.log_param("n_trials",        N_TRIALS)
        mlflow.log_param("n_splits",        N_SPLITS)
        for k, v in best_params.items():
            mlflow.log_param(k, v)
        mlflow.log_artifact(str(out_path), artifact_path="submissions")
    print("MLflow-Logging abgeschlossen.")
