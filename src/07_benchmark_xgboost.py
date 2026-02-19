import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn

from prepare_to_train import get_project_root as get_root_prepare, TARGET_COL, ID_COL
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")

RUN_NAME = "benchmark_xgboost_kaggle_features_fair_split"
RANDOM_STATE = 42


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def create_benchmark_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature-Engineering 1:1 aus dem Kaggle-Beispielnotebook.
    """
    df = df.copy()

    df["study_efficiency"] = df["study_hours_per_day"] / (df["phone_usage_hours"] + 1)
    df["sleep_quality"] = df["sleep_hours"] * df["focus_score"]
    df["distraction_score"] = (
        df["phone_usage_hours"]
        + df["social_media_hours"]
        + df["youtube_hours"]
        + df["gaming_hours"]
    )
    df["productivity_index"] = (
        df["study_hours_per_day"]
        * df["focus_score"]
        * df["attendance_percentage"] / 100
    )
    return df


def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame):
    """
    Gemeinsames LabelEncoding wie im Notebook (train+test gemeinsam fitten).
    """
    train = train.copy()
    test = test.copy()

    cat_cols = train.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        le = LabelEncoder()
        full = pd.concat([train[col], test[col]], axis=0)
        le.fit(full.astype(str))
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

    return train, test


def train_benchmark_with_fair_split():
    project_dir = get_project_root()

    # ------------------------------------------------------
    # 1) Original-Featuredatei laden, um IDs + Split nachzubauen
    # ------------------------------------------------------
    data_features_dir = project_dir / "data" / "features"
    train_pruned_path = data_features_dir / "train_features_pruned.csv"

    train_fe = pd.read_csv(train_pruned_path)

    if ID_COL not in train_fe.columns:
        raise KeyError(f"ID-Spalte '{ID_COL}' nicht in train_features_pruned.csv gefunden.")

    if TARGET_COL not in train_fe.columns:
        raise KeyError(f"Zielvariable '{TARGET_COL}' nicht in train_features_pruned.csv gefunden.")

    # denselben Split wie in load_train_data verwenden
    X_full = train_fe.drop(columns=[TARGET_COL])
    y_full = train_fe[TARGET_COL]

    X_train_fe, X_val_fe, y_train_fe, y_val_fe = train_test_split(
        X_full,
        y_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    train_ids = X_train_fe[ID_COL].values
    val_ids = X_val_fe[ID_COL].values

    # ------------------------------------------------------
    # 2) Rohdaten laden
    # ------------------------------------------------------
    raw_dir = project_dir / "data" / "raw"
    train_raw_path = raw_dir / "train.csv"
    test_raw_path = raw_dir / "test.csv"

    if not train_raw_path.exists() or not test_raw_path.exists():
        raise FileNotFoundError(
            f"Rohdaten nicht gefunden unter {train_raw_path} / {test_raw_path}."
        )

    train_raw = pd.read_csv(train_raw_path)
    test_raw = pd.read_csv(test_raw_path)

    # ------------------------------------------------------
    # 3) Benchmark-Feature-Engineering auf Rohdaten
    # ------------------------------------------------------
    train_raw = create_benchmark_features(train_raw)
    test_raw = create_benchmark_features(test_raw)

    # ------------------------------------------------------
    # 4) Categorical Encoding wie im Notebook (auf train+test)
    # ------------------------------------------------------
    train_raw_enc, test_raw_enc = encode_categoricals(train_raw, test_raw)

    # ------------------------------------------------------
    # 5) Train/Val-Splits in Rohdaten über IDs nachbilden
    # ------------------------------------------------------
    # Sicherstellen, dass ID_COL existiert
    if ID_COL not in train_raw_enc.columns:
        raise KeyError(
            f"ID-Spalte '{ID_COL}' nicht in den Rohdaten gefunden."
        )

    train_part = train_raw_enc[train_raw_enc[ID_COL].isin(train_ids)].copy()
    val_part = train_raw_enc[train_raw_enc[ID_COL].isin(val_ids)].copy()

    # Überprüfen, ob Größe passt
    if len(train_part) != len(train_ids) or len(val_part) != len(val_ids):
        print(
            f"Warnung: Anzahl Zeilen nach ID-Mapping passt nicht exakt – "
            f"Train: {len(train_part)} (IDs: {len(train_ids)}), "
            f"Val: {len(val_part)} (IDs: {len(val_ids)})"
        )

    # ------------------------------------------------------
    # 6) Features/Targets definieren wie im Notebook
    # ------------------------------------------------------
    if TARGET_COL not in train_part.columns:
        raise KeyError(
            f"Zielspalte '{TARGET_COL}' nicht in den Roh-Train-Daten vorhanden."
        )

    X_train = train_part.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    y_train = train_part[TARGET_COL]

    X_val = val_part.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    y_val = val_part[TARGET_COL]

    # Test-Features für evtl. spätere Submission
    X_test = test_raw_enc.drop(columns=[ID_COL], errors="ignore")

    # Align wie im Notebook (auf Nummer sicher):
    X_all = pd.concat([X_train, X_val], axis=0)
    X_all, X_test = X_all.align(X_test, join="left", axis=1, fill_value=0)

    # Nach Align wieder in Train/Val splitten
    X_train = X_all.loc[X_train.index]
    X_val = X_all.loc[X_val.index]

    # ------------------------------------------------------
    # 7) Modell trainieren (ohne KFold, nur Split)
    # ------------------------------------------------------
    mlflow.set_experiment("student_learning_competition")
  #  mlflow.xgboost.autolog()

    with mlflow.start_run(run_name=RUN_NAME):
        model = XGBRegressor(
            n_estimators=1500,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            tree_method="hist",
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=200,
        )

        y_val_pred = model.predict(X_val)
        rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
        r2_val = r2_score(y_val, y_val_pred)

        print(f"[Benchmark XGB] Validation RMSE: {rmse_val:.5f}")
        print(f"[Benchmark XGB] Validation R2:   {r2_val:.5f}")

        mlflow.log_metric("val_rmse", float(rmse_val))
        mlflow.log_metric("val_r2", float(r2_val))

        # Feature Importances
        importances = model.feature_importances_
        fi_df = pd.DataFrame(
            {"feature": X_train.columns, "importance": importances}
        ).sort_values("importance", ascending=False)

        fi_path = project_dir / "models" / "feature_importances_benchmark_xgb.csv"
        fi_path.parent.mkdir(parents=True, exist_ok=True)
        fi_df.to_csv(fi_path, index=False)

        print("\n[Benchmark XGB] Feature Importances:")
        print(fi_df)

        mlflow.log_artifact(str(fi_path), artifact_path="feature_importances")
        mlflow.xgboost.log_model(model, name="benchmark_model")

if __name__ == "__main__":
    train_benchmark_with_fair_split()
