# src/16_optuna_lgbm_light.py
"""
Optuna-Tuning für LightGBM mit zwei Feature-Sets:
  - "benchmark" : exakt die Features des Benchmark-Gewinners (Khoa Tran)
  - "combined"  : Benchmark-Features + engineered Features
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

# ── Konstanten ───────────────────────────────────────────────────────────────────
N_TRIALS     = 50
N_SPLITS     = 5
RANDOM_STATE = 42
MLFLOW_URI   = "http://127.0.0.1:5000"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("student_learning_competition")

BENCHMARK_FEATURES = [
    "age", "study_hours_per_day", "sleep_hours", "phone_usage_hours",
    "social_media_hours", "youtube_hours", "gaming_hours", "breaks_per_day",
    "coffee_intake_mg", "exercise_minutes", "assignments_completed",
    "attendance_percentage", "stress_level", "focus_score", "final_grade",
    "gender",
]

YOUR_ENGINEERED_FEATURES = [
    "screentime_hours", "screentime_to_sleep_ratio", "diligence_score",
    "study_efficiency", "sleep_quality", "productivity_index",
    "study_focus_interaction", "sleep_stress_interaction",
    "attendance_focus_interaction", "distraction_ratio",
    "grade_x_focus", "sleep_minus_phone", "productive_time",
    "study_hours_sq", "focus_score_sq",
]

CAT_FEATURES = ["gender"]


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"social_media_hours", "youtube_hours", "gaming_hours"}.issubset(df.columns):
        df["screentime_hours"] = df["social_media_hours"] + df["youtube_hours"] + df["gaming_hours"]
    if {"screentime_hours", "sleep_hours"}.issubset(df.columns):
        df["screentime_to_sleep_ratio"] = df["screentime_hours"] / (df["sleep_hours"] + 1e-3)
    if {"assignments_completed", "attendance_percentage", "focus_score"}.issubset(df.columns):
        sub = df[["assignments_completed", "attendance_percentage", "focus_score"]].astype(float)
        z = (sub - sub.mean()) / (sub.std(ddof=0) + 1e-9)
        df["diligence_score"] = z.mean(axis=1)
    if {"study_hours_per_day", "phone_usage_hours"}.issubset(df.columns):
        df["study_efficiency"] = df["study_hours_per_day"] / (df["phone_usage_hours"] + 1.0)
    if {"sleep_hours", "focus_score"}.issubset(df.columns):
        df["sleep_quality"] = df["sleep_hours"] * df["focus_score"]
    if {"study_hours_per_day", "focus_score", "attendance_percentage"}.issubset(df.columns):
        df["productivity_index"] = (
            df["study_hours_per_day"] * df["focus_score"] * df["attendance_percentage"] / 100.0
        )
    if {"study_hours_per_day", "focus_score"}.issubset(df.columns):
        df["study_focus_interaction"] = df["study_hours_per_day"] * df["focus_score"]
        df["study_hours_sq"]          = df["study_hours_per_day"] ** 2
        df["focus_score_sq"]          = df["focus_score"] ** 2
    if {"sleep_hours", "stress_level"}.issubset(df.columns):
        df["sleep_stress_interaction"] = df["sleep_hours"] * df["stress_level"]
    if {"attendance_percentage", "focus_score"}.issubset(df.columns):
        df["attendance_focus_interaction"] = df["attendance_percentage"] * df["focus_score"]
    if {"final_grade", "focus_score"}.issubset(df.columns):
        df["grade_x_focus"] = df["final_grade"] * df["focus_score"]
    if {"sleep_hours", "phone_usage_hours"}.issubset(df.columns):
        df["sleep_minus_phone"] = df["sleep_hours"] - df["phone_usage_hours"]
    if {"study_hours_per_day", "sleep_hours", "phone_usage_hours"}.issubset(df.columns):
        df["productive_time"] = (
            df["study_hours_per_day"] + df["sleep_hours"] - df["phone_usage_hours"]
        )
    if {"social_media_hours", "youtube_hours", "gaming_hours",
        "phone_usage_hours", "study_hours_per_day"}.issubset(df.columns):
        total_dist = (df["social_media_hours"] + df["youtube_hours"]
                      + df["gaming_hours"] + df["phone_usage_hours"])
        df["distraction_ratio"] = total_dist / (df["study_hours_per_day"] + 1e-3)
    return df


def load_feature_sets(project_dir: Path) -> dict:
    train_path, test_path = None, None
    for subdir in ["raw", "processed"]:
        tp = project_dir / "data" / subdir / "train.csv"
        vp = project_dir / "data" / subdir / "test.csv"
        if tp.exists():
            train_path, test_path = tp, vp
            break
    if train_path is None:
        train_path = project_dir / "data" / "processed" / "train_clean.csv"
        test_path  = project_dir / "data" / "processed" / "test_clean.csv"

    print(f"  Lade Daten aus: {train_path.parent}")
    train_raw = pd.read_csv(train_path)
    test_raw  = pd.read_csv(test_path)

    for df in [train_raw, test_raw]:
        if "student_id" in df.columns:
            df.drop(columns=["student_id"], inplace=True)

    TARGET = "productivity_score"
    ID_COL = "id"

    y        = train_raw[TARGET].values
    test_ids = test_raw[ID_COL].values if ID_COL in test_raw.columns else np.arange(len(test_raw))

    # Feature-Set 1: Benchmark
    bench_cols   = [c for c in BENCHMARK_FEATURES if c in train_raw.columns]
    X_bench      = train_raw[bench_cols].copy()
    X_test_bench = test_raw[bench_cols].copy()
    for col in CAT_FEATURES:
        if col in X_bench.columns:
            X_bench[col]      = X_bench[col].astype("category")
            X_test_bench[col] = X_test_bench[col].astype("category")

    # Feature-Set 2: Combined
    train_eng       = add_interactions(train_raw)
    test_eng        = add_interactions(test_raw)
    eng_cols        = [c for c in YOUR_ENGINEERED_FEATURES if c in train_eng.columns]
    combined_cols   = bench_cols + [c for c in eng_cols if c not in bench_cols]
    X_combined      = train_eng[combined_cols].copy()
    X_test_combined = test_eng[combined_cols].copy()
    for col in CAT_FEATURES:
        if col in X_combined.columns:
            X_combined[col]      = X_combined[col].astype("category")
            X_test_combined[col] = X_test_combined[col].astype("category")

    print(f"  Benchmark-Features ({len(bench_cols)}): {bench_cols}")
    print(f"  Combined-Features  ({len(combined_cols)}): {combined_cols}")

    return {
        "benchmark": {
            "X_train": X_bench, "X_test": X_test_bench,
            "y_train": y, "test_ids": test_ids,
            "cat_cols": CAT_FEATURES, "feature_names": bench_cols,
        },
        "combined": {
            "X_train": X_combined, "X_test": X_test_combined,
            "y_train": y, "test_ids": test_ids,
            "cat_cols": CAT_FEATURES, "feature_names": combined_cols,
        },
    }


# ── Optuna Objective ──────────────────────────────────────────────────────────────

def objective(trial, X, y, cat_cols):
    params = {
        "objective"        : "regression",
        "metric"           : "rmse",
        "verbosity"        : -1,
        "n_jobs"           : 2,
        "random_state"     : RANDOM_STATE,
        "n_estimators"     : 3000,
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
    kf     = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr = X.iloc[train_idx].copy()
        X_v  = X.iloc[val_idx].copy()
        y_tr = y[train_idx]
        y_v  = y[val_idx]
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            categorical_feature=cat_cols if cat_cols else "auto",
            eval_set=[(X_v, y_v)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        scores.append(rmse(y_v, model.predict(X_v)))
    return float(np.mean(scores))


# ── Finales Modell trainieren ─────────────────────────────────────────────────────

def train_final_model(label, best_params, X, y, X_test, cat_cols, test_ids, project_dir):
    print(f"\n[{label}] Trainiere finales Modell (KFold OOF) ...")
    final_params = {
        "objective"   : "regression",
        "metric"      : "rmse",
        "verbosity"   : -1,
        "n_jobs"      : 2,
        "random_state": RANDOM_STATE,
        "n_estimators": 3000,
        **best_params,
    }
    kf             = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
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
            categorical_feature=cat_cols if cat_cols else "auto",
            eval_set=[(X_v, y_v)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        fold_oof  = model.predict(X_v)
        fold_test = model.predict(X_test)
        oof_preds[val_idx] = fold_oof
        test_preds_all.append(fold_test)
        print(f"RMSE = {rmse(y_v, fold_oof):.5f}")

    oof_rmse   = rmse(y, oof_preds)
    test_preds = np.mean(test_preds_all, axis=0)
    print(f"  OOF-RMSE gesamt: {oof_rmse:.5f}")

    # Submission speichern
    out_path = project_dir / "models" / f"submission_lgbm_{label}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "id": test_ids,
        "productivity_score": test_preds,
    }).to_csv(out_path, index=False)
    print(f"  Submission gespeichert: {out_path}")

    return oof_preds, test_preds, oof_rmse


# ── Hauptprogramm ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[1]
    print("Lade Feature-Sets ...")
    feature_sets = load_feature_sets(project_dir)

    all_results = {}

    for label, fs in feature_sets.items():
        X        = fs["X_train"]
        y        = fs["y_train"]
        X_test   = fs["X_test"]
        cat_cols = fs["cat_cols"]
        test_ids = fs["test_ids"]

        print(f"\n{'=' * 60}")
        print(f"  FEATURE-SET: {label.upper()}  ({X.shape[1]} Features)")
        print(f"{'=' * 60}")

        # ── Optuna-Tuning ──────────────────────────────────────────────────────
        print(f"\n[{label}] Starte Optuna ({N_TRIALS} Trials) ...")
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(
            lambda trial: objective(trial, X, y, cat_cols),
            n_trials=N_TRIALS,
            n_jobs=1,
            show_progress_bar=True,
        )
        best_params  = study.best_params
        best_cv_rmse = study.best_value
        print(f"\n  Beste Parameter : {best_params}")
        print(f"  Bester CV-RMSE  : {best_cv_rmse:.5f}")

        # ── Finales Modell ─────────────────────────────────────────────────────
        oof_preds, test_preds, oof_rmse = train_final_model(
            label, best_params, X, y, X_test, cat_cols, test_ids, project_dir
        )

        all_results[label] = {
            "best_params": best_params,
            "cv_rmse"    : best_cv_rmse,
            "oof_rmse"   : oof_rmse,
            "oof_preds"  : oof_preds,
            "test_preds" : test_preds,
        }

        # ── MLflow-Logging ─────────────────────────────────────────────────────
        with mlflow.start_run(run_name=f"lgbm_optuna_{label}"):
            mlflow.log_metric("optuna_cv_rmse", best_cv_rmse)
            mlflow.log_metric("oof_rmse",       oof_rmse)
            mlflow.log_param("feature_set",     label)
            mlflow.log_param("n_features",      X.shape[1])
            mlflow.log_param("n_trials",        N_TRIALS)
            mlflow.log_param("n_splits",        N_SPLITS)
            for k, v in best_params.items():
                mlflow.log_param(k, v)
            mlflow.log_artifact(
                str(project_dir / "models" / f"submission_lgbm_{label}.csv"),
                artifact_path="submissions",
            )
        print(f"  MLflow-Run 'lgbm_optuna_{label}' abgeschlossen.")

    # ── Zusammenfassung ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  ZUSAMMENFASSUNG")
    print(f"{'=' * 60}")
    for label, res in all_results.items():
        print(f"\n  [{label}]")
        print(f"    Optuna CV-RMSE : {res['cv_rmse']:.5f}")
        print(f"    OOF-RMSE       : {res['oof_rmse']:.5f}")
        print(f"    Beste Parameter: {res['best_params']}")

    # ── Ensemble-Submission ────────────────────────────────────────────────────
    if len(all_results) == 2:
        r_b = all_results["benchmark"]
        r_c = all_results["combined"]

        w_b = 1.0 / r_b["oof_rmse"]
        w_c = 1.0 / r_c["oof_rmse"]
        w_total = w_b + w_c
        w_b /= w_total
        w_c /= w_total

        print(f"\n  Ensemble-Gewichte:")
        print(f"    benchmark : {w_b:.4f}  (OOF-RMSE = {r_b['oof_rmse']:.5f})")
        print(f"    combined  : {w_c:.4f}  (OOF-RMSE = {r_c['oof_rmse']:.5f})")

        ensemble_preds = w_b * r_b["test_preds"] + w_c * r_c["test_preds"]

        ens_path = project_dir / "models" / "submission_lgbm_ensemble.csv"
        pd.DataFrame({
            "id"                : feature_sets["benchmark"]["test_ids"],
            "productivity_score": ensemble_preds,
        }).to_csv(ens_path, index=False)
        print(f"\n  Ensemble-Submission gespeichert: {ens_path}")

        with mlflow.start_run(run_name="lgbm_optuna_ensemble"):
            mlflow.log_metric("weight_benchmark",   w_b)
            mlflow.log_metric("weight_combined",    w_c)
            mlflow.log_metric("oof_rmse_benchmark", r_b["oof_rmse"])
            mlflow.log_metric("oof_rmse_combined",  r_c["oof_rmse"])
            mlflow.log_artifact(str(ens_path), artifact_path="submissions")
        print("  MLflow-Ensemble-Run abgeschlossen.")

    print(f"\nFertig. Submissions in: {project_dir / 'models'}")
    print("  -> submission_lgbm_benchmark.csv")
    print("  -> submission_lgbm_combined.csv")
    print("  -> submission_lgbm_ensemble.csv  (empfohlen)")

