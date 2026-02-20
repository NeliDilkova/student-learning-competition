# src/15_spline_elasticnet_experiment.py
"""
Spline-Transformer + ElasticNet Experiment
==========================================
Ziel: Lineare Modelle mit Spline-Features testen, da der Benchmark (LGBM, 0.037xx)
      bereits andeutete, dass die Top-6-Features "surprisingly linear signals" sind.

Pipeline:
  1. SplineTransformer (für nicht-lineare Zusammenhänge)
  2. StandardScaler (für ElasticNet-Regularisierung)
  3. ElasticNet (L1+L2)

Getestet auf zwei Feature-Sets:
  - "raw"       : Nur Original-Features (wie Benchmark)
  - "engineered": + Feature-Engineering aus 03_features.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor

import lightgbm as lgb
from catboost import CatBoostRegressor, Pool

# ── Konstanten ──────────────────────────────────────────────────────────────────
N_OPTUNA_TRIALS   = 30
N_SPLITS          = 3
RANDOM_STATE      = 42
MLFLOW_URI        = "http://127.0.0.1:5000"
EXPERIMENT_NAME   = "student_learning_competition"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# ── Hilfsfunktionen ─────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def kfold_cv_rmse(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    """Gibt (mean_rmse, std_rmse) zurück."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_v = X[train_idx], X[val_idx]
        y_tr, y_v = y[train_idx], y[val_idx]
        model.fit(X_tr, y_tr)
        scores.append(rmse(y_v, model.predict(X_v)))
    return float(np.mean(scores)), float(np.std(scores))


# ── 1. Basis-Vergleich ──────────────────────────────────────────────────────────

def baseline_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    label: str,
) -> dict:
    """
    Vergleicht mehrere einfache Baseline-Modelle via 5-Fold-CV.
    Gibt ein dict {model_name: mean_rmse} zurück.
    """
    print(f"\n[{label}] ── Baseline-Vergleich ──────────────────────────────")

    models = {
        "DummyMean"    : DummyRegressor(strategy="mean"),
        "Ridge"        : Ridge(alpha=1.0),
        "ElasticNet"   : ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
        "SplineRidge"  : make_pipeline(
                            SplineTransformer(n_knots=5, degree=3, include_bias=False),
                            StandardScaler(),
                            Ridge(alpha=1.0),
                         ),
        "SplineElastic": make_pipeline(
                            SplineTransformer(n_knots=5, degree=3, include_bias=False),
                            StandardScaler(),
                            ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
                         ),
    }

    results = {}
    for name, model in models.items():
        mean_r, std_r = kfold_cv_rmse(model, X_train, y_train)
        results[name] = mean_r
        print(f"  {name:<20s}  RMSE = {mean_r:.5f} ± {std_r:.5f}")

    return results


# ── 2. Optuna-Tuning ────────────────────────────────────────────────────────────

def tune_spline_elasticnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    label: str,
    n_trials: int = N_OPTUNA_TRIALS,
) -> tuple[dict, float, optuna.Study]:
    """
    Optimiert SplineTransformer + StandardScaler + ElasticNet via Optuna.
    Gibt (best_params, best_cv_rmse, study) zurück.
    """
    print(f"\n[{label}] ── Optuna-Tuning ({n_trials} Trials) ────────────────")

    def objective(trial: optuna.Trial) -> float:
        n_knots  = trial.suggest_int("n_knots",   3, 6)      # max 6 statt 10
        degree   = trial.suggest_int("degree",    2, 3)      # max 3 statt 5
        alpha    = trial.suggest_float("alpha",   1e-3, 5.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

        model = make_pipeline(
            SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False),
            StandardScaler(),
            ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5_000),
        )
        # KEIN n_jobs — sequenziell, ein Fold nach dem anderen
        mean_r, _ = kfold_cv_rmse(model, X_train, y_train, n_splits=N_SPLITS)
        return mean_r

    # n_jobs=1 → sequenziell, kein Multiprocessing-Crash
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,           # ← KRITISCH: kein Parallel-Tuning
    )

    best_params  = study.best_params
    best_cv_rmse = study.best_value
    print(f"  Beste Parameter : {best_params}")
    print(f"  Bester CV-RMSE  : {best_cv_rmse:.5f}")
    return best_params, best_cv_rmse, study

# ── 3. Finales Modell & OOF-Predictions sequenziell ────────────────────────────────────────

def fit_final_model(
    best_params: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
) -> tuple[object, np.ndarray, np.ndarray, float]:
    n_samples = X_train.shape[0]
    oof_preds = np.zeros(n_samples)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_test_preds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"    Fold {fold+1}/{N_SPLITS} ...", end=" ", flush=True)
        X_tr, X_v = X_train[train_idx], X_train[val_idx]
        y_tr, y_v = y_train[train_idx], y_train[val_idx]

        model = make_pipeline(
            SplineTransformer(
                n_knots      = best_params["n_knots"],
                degree       = best_params["degree"],
                include_bias = False,
            ),
            StandardScaler(),
            ElasticNet(
                alpha    = best_params["alpha"],
                l1_ratio = best_params["l1_ratio"],
                max_iter = 5_000,
            ),
        )
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict(X_v)
        fold_test_preds.append(model.predict(X_test))
        print(f"RMSE = {rmse(y_v, model.predict(X_v)):.5f}")

    oof_rmse   = rmse(y_train, oof_preds)
    test_preds = np.mean(fold_test_preds, axis=0)

    # Finales Modell auf allen Trainingsdaten
    final_model = make_pipeline(
        SplineTransformer(
            n_knots      = best_params["n_knots"],
            degree       = best_params["degree"],
            include_bias = False,
        ),
        StandardScaler(),
        ElasticNet(
            alpha    = best_params["alpha"],
            l1_ratio = best_params["l1_ratio"],
            max_iter = 5_000,
        ),
    )
    final_model.fit(X_train, y_train)
    print(f"  OOF-RMSE: {oof_rmse:.5f}")
    return final_model, oof_preds, test_preds, oof_rmse


# ── Baseline ebenfalls ohne Parallelisierung ────────────────────────────────────
def baseline_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    label: str,
) -> dict:
    print(f"\n[{label}] ── Baseline-Vergleich ──────────────────────────────")

    models = {
        "DummyMean"    : DummyRegressor(strategy="mean"),
        "Ridge"        : Ridge(alpha=1.0),
        "ElasticNet"   : ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
        "SplineRidge"  : make_pipeline(
                            SplineTransformer(n_knots=5, degree=3, include_bias=False),
                            StandardScaler(),
                            Ridge(alpha=1.0),
                         ),
        "SplineElastic": make_pipeline(
                            SplineTransformer(n_knots=5, degree=3, include_bias=False),
                            StandardScaler(),
                            ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
                         ),
    }

    results = {}
    for name, model in models.items():
        # n_splits=3 und sequenziell
        mean_r, std_r = kfold_cv_rmse(model, X_train, y_train, n_splits=N_SPLITS)
        results[name] = mean_r
        print(f"  {name:<20s}  RMSE = {mean_r:.5f} ± {std_r:.5f}")

    return results

# ── 4. Submission speichern ─────────────────────────────────────────────────────

def save_submission(
    test_ids: np.ndarray,
    test_preds: np.ndarray,
    output_path: Path,
) -> None:
    submission = pd.DataFrame({
        "id"                : test_ids,
        "productivity_score": test_preds,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"  Submission gespeichert: {output_path}")


# ── Run pro Feature-Set ────────────────────────────────────────────────────────
def run_feature_set(
    label: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    test_ids: np.ndarray,
    feature_names: list,
    n_trials: int = N_OPTUNA_TRIALS,
) -> dict:
    """
    Vollständiger Durchlauf für ein Feature-Set:
      1. Basis-Vergleich
      2. Optuna-Tuning
      3. Finales Modell (KFold auf gesamten Trainingsdaten)
      4. Submission speichern
      5. MLflow-Logging
    """
    print(f"\n{'=' * 60}")
    print(f"  FEATURE-SET: {label.upper()}  ({X_train.shape[1]} Features)")
    print(f"{'=' * 60}")

    # 1. Basis-Vergleich
    baseline_results = baseline_comparison(X_train, y_train, label)

    # 2. Optuna-Tuning
    best_params, best_cv_rmse, study = tune_spline_elasticnet(
        X_train, y_train, label, n_trials=n_trials
    )

    # 3. Finales Modell mit besten Parametern
    final_model, oof_preds, test_preds, oof_rmse = fit_final_model(
        best_params, X_train, X_test, y_train
    )

    # 4. Submission speichern
    project_dir = Path(__file__).resolve().parents[1]
    submission_path = project_dir / "models" / f"submission_spline_{label}.csv"
    save_submission(test_ids, test_preds, submission_path)

    # 5. MLflow-Logging
    run_name = f"spline_elasticnet_{label}"
    with mlflow.start_run(run_name=run_name):
        # Parameter
        mlflow.log_param("feature_set",   label)
        mlflow.log_param("n_features",    X_train.shape[1])
        mlflow.log_param("n_optuna_trials", n_trials)
        mlflow.log_param("n_splits",      N_SPLITS)
        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", v)

        # Metriken
        mlflow.log_metric("optuna_cv_rmse", best_cv_rmse)
        mlflow.log_metric("oof_rmse",       oof_rmse)

        # Baseline-Metriken
        for model_name, bl_rmse in baseline_results.items():
            mlflow.log_metric(f"baseline_{model_name}_rmse", bl_rmse)

        # Feature-Namen als Artifact speichern
        feat_path = project_dir / "models" / f"features_{label}.txt"
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        feat_path.write_text("\n".join(feature_names))
        mlflow.log_artifact(str(feat_path), artifact_path="features")

        # Submission-Artifact
        mlflow.log_artifact(str(submission_path), artifact_path="submissions")

        # Modell loggen
        mlflow.sklearn.log_model(final_model, artifact_path="model")

        print(f"\n[{label}] MLflow Run abgeschlossen.")
        print(f"  Optuna CV-RMSE : {best_cv_rmse:.5f}")
        print(f"  OOF-RMSE       : {oof_rmse:.5f}")

    return {
        "label"          : label,
        "best_params"    : best_params,
        "optuna_cv_rmse" : best_cv_rmse,
        "oof_rmse"       : oof_rmse,
        "oof_preds"      : oof_preds,
        "test_preds"     : test_preds,
        "final_model"    : final_model,
        "baseline"       : baseline_results,
    }


# ── Daten laden & Feature Engineering ──────────────────────────────────────────

def load_raw_data(project_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lädt train_features.csv und test_features.csv aus data/features/."""
    features_dir = project_dir / "data" / "features"
    train = pd.read_csv(features_dir / "train_features.csv")
    test  = pd.read_csv(features_dir / "test_features.csv")
    return train, test


def build_feature_sets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "productivity_score",
    id_col: str = "id",
) -> dict:
    """
    Baut zwei Feature-Sets:
      - "raw"       : Nur Original-Features (keine engineered Spalten)
      - "engineered": Alle verfügbaren Features aus train_features.csv

    Gibt ein dict zurück:
      {
        "raw": {
            "X_train", "X_test", "y_train", "test_ids", "feature_names"
        },
        "engineered": { ... }
      }
    """
    # Zielvariable & ID trennen
    y_train  = train[target_col].values
    test_ids = test[id_col].values if id_col in test.columns else np.arange(len(test))

    # Original (rohe) Features — identisch mit dem Benchmark
    raw_feature_names = [
        "study_hours_per_day",
        "sleep_hours",
        "phone_usage_hours",
        "social_media_hours",
        "youtube_hours",
        "gaming_hours",
        "breaks_per_day",
        "coffee_intake_mg",
        "exercise_minutes",
        "assignments_completed",
        "attendance_percentage",
        "stress_level",
        "focus_score",
        "final_grade",
    ]
    # Nur Spalten behalten, die tatsächlich im DataFrame vorhanden sind
    raw_feature_names = [c for c in raw_feature_names if c in train.columns]

    # Engineered Feature-Namen: gender wird one-hot kodiert, Bins werden numerisch
    # Alle numerischen Spalten außer Ziel und ID
    drop_cols = {target_col, id_col}
    all_num_cols = [
        c for c in train.select_dtypes(include=["int64", "float64", "float32"]).columns
        if c not in drop_cols
    ]

    # Kategoriale Spalten one-hot kodieren (gender, age_bin, etc.)
    cat_cols = [
        c for c in train.select_dtypes(include=["object", "category"]).columns
        if c not in drop_cols
    ]

    # One-Hot Encoding für beide Sets konsistent
    train_enc = pd.get_dummies(train, columns=cat_cols, drop_first=False)
    test_enc  = pd.get_dummies(test,  columns=cat_cols, drop_first=False)

    # Spalten angleichen (Test kann weniger Kategorien haben)
    train_enc, test_enc = train_enc.align(test_enc, join="left", axis=1, fill_value=0)

    eng_feature_names = [
        c for c in train_enc.columns
        if c not in drop_cols
    ]

    def to_numpy(df: pd.DataFrame, cols: list) -> np.ndarray:
        """Gibt einen float64-Array zurück; fehlende Spalten → 0."""
        present = [c for c in cols if c in df.columns]
        arr = df[present].fillna(0).astype(float).values
        return arr

    # Zusätzliche Interaktions-Features, die im Benchmark nicht vorhanden sind
    def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Kern-Interaktionen
        if {"study_hours_per_day", "focus_score"}.issubset(df.columns):
            df["study_x_focus"]         = df["study_hours_per_day"] * df["focus_score"]
            df["study_hours_sq"]        = df["study_hours_per_day"] ** 2
            df["focus_score_sq"]        = df["focus_score"] ** 2
        if {"attendance_percentage", "focus_score"}.issubset(df.columns):
            df["attendance_x_focus"]    = df["attendance_percentage"] * df["focus_score"] / 100.0
        if {"sleep_hours", "phone_usage_hours"}.issubset(df.columns):
            df["sleep_minus_phone"]     = df["sleep_hours"] - df["phone_usage_hours"]
        if {"study_hours_per_day", "sleep_hours", "phone_usage_hours"}.issubset(df.columns):
            df["productive_time"]       = (df["study_hours_per_day"]
                                           + df["sleep_hours"]
                                           - df["phone_usage_hours"])
        if {"social_media_hours", "youtube_hours", "gaming_hours",
            "phone_usage_hours", "study_hours_per_day"}.issubset(df.columns):
            total_distraction = (df["social_media_hours"]
                                 + df["youtube_hours"]
                                 + df["gaming_hours"]
                                 + df["phone_usage_hours"])
            df["distraction_ratio"] = total_distraction / (df["study_hours_per_day"] + 1e-3)
        if {"final_grade", "focus_score"}.issubset(df.columns):
            df["grade_x_focus"]         = df["final_grade"] * df["focus_score"]
        if {"stress_level", "sleep_hours"}.issubset(df.columns):
            df["stress_x_sleep"]        = df["stress_level"] * df["sleep_hours"]
        if {"assignments_completed", "attendance_percentage"}.issubset(df.columns):
            df["assign_x_attendance"]   = (df["assignments_completed"]
                                           * df["attendance_percentage"] / 100.0)
        return df

    train_with_inter = add_interactions(train_enc)
    test_with_inter  = add_interactions(test_enc)

    # Spalten erneut angleichen nach Interaktionen
    train_with_inter, test_with_inter = train_with_inter.align(
        test_with_inter, join="left", axis=1, fill_value=0
    )

    eng_plus_inter_names = [
        c for c in train_with_inter.columns
        if c not in drop_cols
    ]

    return {
        "raw": {
            "X_train"      : to_numpy(train,            raw_feature_names),
            "X_test"       : to_numpy(test,             raw_feature_names),
            "y_train"      : y_train,
            "test_ids"     : test_ids,
            "feature_names": raw_feature_names,
        },
        "engineered": {
            "X_train"      : to_numpy(train_with_inter, eng_plus_inter_names),
            "X_test"       : to_numpy(test_with_inter,  eng_plus_inter_names),
            "y_train"      : y_train,
            "test_ids"     : test_ids,
            "feature_names": eng_plus_inter_names,
        },
    }


# ── Ensemble aus beiden Runs ────────────────────────────────────────────────────

def blend_submissions(
    results: dict[str, dict],
    test_ids: np.ndarray,
    project_dir: Path,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Erstellt eine gewichtete Ensemble-Submission aus allen Run-Ergebnissen.
    Gewichte basieren standardmäßig auf 1/oof_rmse (besser → höheres Gewicht).
    """
    if weights is None:
        # Automatisch: Gewicht ∝ 1 / OOF-RMSE
        raw_weights = {label: 1.0 / res["oof_rmse"] for label, res in results.items()}
        total = sum(raw_weights.values())
        weights = {k: v / total for k, v in raw_weights.items()}

    print("\n── Ensemble-Gewichte ──────────────────────────────────────────")
    for label, w in weights.items():
        print(f"  {label:<20s}  Gewicht = {w:.4f}  "
              f"(OOF-RMSE = {results[label]['oof_rmse']:.5f})")

    # Gewichtete Summe der Test-Predictions
    blended = np.zeros(len(test_ids))
    for label, res in results.items():
        blended += weights[label] * res["test_preds"]

    submission = pd.DataFrame({
        "id"               : test_ids,
        "productivity_score": blended,
    })

    out_path = project_dir / "models" / "submission_spline_ensemble.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"\n  Ensemble-Submission gespeichert: {out_path}")

    # MLflow-Logging für Ensemble
    with mlflow.start_run(run_name="spline_elasticnet_ensemble"):
        for label, w in weights.items():
            mlflow.log_metric(f"weight_{label}",    w)
            mlflow.log_metric(f"oof_rmse_{label}",  results[label]["oof_rmse"])
        mlflow.log_artifact(str(out_path), artifact_path="submissions")

    return submission


# ── Hauptprogramm ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[1]

    print("Lade Daten ...")
    train_df, test_df = load_raw_data(project_dir)
    print(f"  Train: {train_df.shape}  |  Test: {test_df.shape}")

    print("\nBaue Feature-Sets ...")
    feature_sets = build_feature_sets(train_df, test_df)
    for fs_label, fs_data in feature_sets.items():
        print(f"  [{fs_label}]  X_train: {fs_data['X_train'].shape}  "
              f"X_test: {fs_data['X_test'].shape}")

    # ── Runs für beide Feature-Sets ──────────────────────────────────────────────
    all_results: dict[str, dict] = {}

    for fs_label, fs_data in feature_sets.items():
        result = run_feature_set(
            label        = fs_label,
            X_train      = fs_data["X_train"],
            X_test       = fs_data["X_test"],
            y_train      = fs_data["y_train"],
            test_ids     = fs_data["test_ids"],
            feature_names= fs_data["feature_names"],
            n_trials     = N_OPTUNA_TRIALS,
        )
        all_results[fs_label] = result

    # ── Zusammenfassung ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ZUSAMMENFASSUNG ALLER RUNS")
    print("=" * 60)
    for label, res in all_results.items():
        print(f"\n  [{label}]")
        print(f"    Optuna CV-RMSE : {res['optuna_cv_rmse']:.5f}")
        print(f"    OOF-RMSE       : {res['oof_rmse']:.5f}")
        print(f"    Beste Parameter: {res['best_params']}")

    # ── Ensemble-Submission ──────────────────────────────────────────────────────
    # test_ids ist in beiden Feature-Sets identisch
    test_ids = next(iter(feature_sets.values()))["test_ids"]

    if len(all_results) > 1:
        ensemble_submission = blend_submissions(
            results    = all_results,
            test_ids   = test_ids,
            project_dir= project_dir,
        )
        print("\nEnsemble-Submission (erste Zeilen):")
        print(ensemble_submission.head())
    else:
        # Nur ein Feature-Set → direkt die einzelne Submission verwenden
        only_result = next(iter(all_results.values()))
        print(f"\nNur ein Feature-Set vorhanden — "
              f"Submission bereits gespeichert (OOF-RMSE: {only_result['oof_rmse']:.5f})")

    print("\nFertig. Alle Submissions liegen unter:", project_dir / "models")

