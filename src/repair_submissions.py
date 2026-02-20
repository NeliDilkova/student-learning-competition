# src/17_fix_submission.py
"""
Erstellt korrekte Submission mit den echten Kaggle-Testdaten.
Verwendet absolute Pfade und bewährte Benchmark-Parameter (Khoa Tran).
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import mlflow

# ── PFADE EXPLIZIT SETZEN ────────────────────────────────────────────────────────
RAW_DATA_DIR = Path(r"C:\Users\nelid\Documents\Kaggle Competitions\Student Learning Competition\student-learning-competition\data\raw")
OUTPUT_DIR   = Path(r"C:\Users\nelid\Documents\Kaggle Competitions\Student Learning Competition\student-learning-competition\models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_URI   = "http://127.0.0.1:5000"
N_SPLITS     = 5
RANDOM_STATE = 42

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("student_learning_competition")


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


if __name__ == "__main__":

    # ── Daten laden ──────────────────────────────────────────────────────────────
    print(f"Lade Daten aus: {RAW_DATA_DIR}")
    train_raw = pd.read_csv(RAW_DATA_DIR / "train.csv")
    test_raw  = pd.read_csv(RAW_DATA_DIR / "test.csv")
    sample    = pd.read_csv(RAW_DATA_DIR / "sample_submission.csv")

    print(f"Train: {train_raw.shape} | Test: {test_raw.shape}")
    print(f"Sample submission IDs (erste 5): {sample['id'].head().tolist()}")
    print(f"Test IDs (erste 5):              {test_raw['id'].head().tolist()}")

    # student_id droppen
    for df in [train_raw, test_raw]:
        if "student_id" in df.columns:
            df.drop(columns=["student_id"], inplace=True)

    TARGET   = "productivity_score"
    ID_COL   = "id"
    CAT_COLS = ["gender"]

    y        = train_raw[TARGET].values
    test_ids = test_raw[ID_COL].values

    # ── Feature-Sets ─────────────────────────────────────────────────────────────
    BENCHMARK_COLS = [
        "age", "study_hours_per_day", "sleep_hours", "phone_usage_hours",
        "social_media_hours", "youtube_hours", "gaming_hours", "breaks_per_day",
        "coffee_intake_mg", "exercise_minutes", "assignments_completed",
        "attendance_percentage", "stress_level", "focus_score", "final_grade",
        "gender",
    ]
    bench_cols = [c for c in BENCHMARK_COLS if c in train_raw.columns]

    train_eng = add_interactions(train_raw)
    test_eng  = add_interactions(test_raw)

    ENG_COLS = [
        "screentime_hours", "screentime_to_sleep_ratio", "diligence_score",
        "study_efficiency", "sleep_quality", "productivity_index",
        "study_focus_interaction", "sleep_stress_interaction",
        "attendance_focus_interaction", "distraction_ratio",
        "grade_x_focus", "sleep_minus_phone", "productive_time",
        "study_hours_sq", "focus_score_sq",
    ]
    combined_cols = bench_cols + [c for c in ENG_COLS if c in train_eng.columns and c not in bench_cols]

    feature_sets = {
        "benchmark": (train_raw[bench_cols].copy(),    test_raw[bench_cols].copy()),
        "combined" : (train_eng[combined_cols].copy(), test_eng[combined_cols].copy()),
    }

    for label, (X, X_test) in feature_sets.items():
        for col in CAT_COLS:
            if col in X.columns:
                X[col]      = X[col].astype("category")
                X_test[col] = X_test[col].astype("category")
        feature_sets[label] = (X, X_test)

    print(f"\nBenchmark Features: {len(bench_cols)}")
    print(f"Combined Features:  {len(combined_cols)}")

    # ── Beste Parameter (Khoa Tran Benchmark) ────────────────────────────────────
    BEST_PARAMS_BENCHMARK = {
        "learning_rate"    : 0.4519543556274892,
        "max_depth"        : 4,
        "num_leaves"       : 4,
        "min_child_samples": 27,
        "subsample"        : 0.8830822978218495,
        "subsample_freq"   : 7,
        "colsample_bytree" : 0.6394789787333435,
        "reg_alpha"        : 2.8227813683691134,
        "reg_lambda"       : 3.609134307913427,
    }
    BEST_PARAMS_COMBINED = BEST_PARAMS_BENCHMARK.copy()

    best_params_map = {
        "benchmark": BEST_PARAMS_BENCHMARK,
        "combined" : BEST_PARAMS_COMBINED,
    }

    # ── Training & Submission ────────────────────────────────────────────────────
    all_test_preds = {}
    all_oof_rmse   = {}

    for label, (X, X_test) in feature_sets.items():
        print(f"\n{'=' * 50}")
        print(f"  {label.upper()}  ({X.shape[1]} Features)")
        print(f"{'=' * 50}")

        params = {
            "objective"   : "regression",
            "metric"      : "rmse",
            "verbosity"   : -1,
            "n_jobs"      : 2,
            "random_state": RANDOM_STATE,
            "n_estimators": 4200,
            **best_params_map[label],
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

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                categorical_feature=CAT_COLS,
                eval_set=[(X_v, y_v)],
                callbacks=[
                    lgb.early_stopping(100, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            fold_oof  = model.predict(X_v)
            fold_test = model.predict(X_test)
            oof_preds[val_idx] = fold_oof
            test_preds_all.append(fold_test)
            print(f"RMSE = {rmse(y_v, fold_oof):.5f}")

        oof_rmse   = rmse(y, oof_preds)
        test_preds = np.mean(test_preds_all, axis=0)
        all_oof_rmse[label]   = oof_rmse
        all_test_preds[label] = test_preds
        print(f"  OOF-RMSE: {oof_rmse:.5f}")

        out_path = OUTPUT_DIR / f"submission_v2_{label}.csv"
        pd.DataFrame({
            "id"                : test_ids,
            "productivity_score": test_preds,
        }).to_csv(out_path, index=False)
        print(f"  Gespeichert: {out_path}")

    # ── Ensemble ─────────────────────────────────────────────────────────────────
    w_b     = 1.0 / all_oof_rmse["benchmark"]
    w_c     = 1.0 / all_oof_rmse["combined"]
    w_total = w_b + w_c

    ensemble_preds = (
        (w_b / w_total) * all_test_preds["benchmark"] +
        (w_c / w_total) * all_test_preds["combined"]
    )

    ens_path = OUTPUT_DIR / "submission_v2_ensemble.csv"
    pd.DataFrame({
        "id"                : test_ids,
        "productivity_score": ensemble_preds,
    }).to_csv(ens_path, index=False)

    print(f"\n{'=' * 50}")
    print("  ZUSAMMENFASSUNG")
    print(f"{'=' * 50}")
    print(f"  benchmark OOF-RMSE : {all_oof_rmse['benchmark']:.5f}")
    print(f"  combined  OOF-RMSE : {all_oof_rmse['combined']:.5f}")
    print(f"  Ensemble gespeichert: {ens_path}")

    # ── MLflow ────────────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="lgbm_v2_fixed_paths"):
        for label, r in all_oof_rmse.items():
            mlflow.log_metric(f"oof_rmse_{label}", r)
        mlflow.log_metric("weight_benchmark", w_b / w_total)
        mlflow.log_metric("weight_combined",  w_c / w_total)
        mlflow.log_artifact(str(ens_path), artifact_path="submissions")
    print("  MLflow-Logging abgeschlossen.")

    print("\nFertig. Einreichen:")
    print(f"  -> {ens_path}")
