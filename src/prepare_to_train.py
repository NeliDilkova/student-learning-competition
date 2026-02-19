# src/05_prepare_to_train.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COL = "productivity_score"
ID_COL = "id"


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_train_data(test_size: float = 0.2, random_state: int = 42):
    project_dir = get_project_root()
    data_features_dir = project_dir / "data" / "features"

    train_pruned_path = data_features_dir / "train_features_pruned.csv"
    if not train_pruned_path.exists():
        raise FileNotFoundError(
            f"{train_pruned_path} nicht gefunden. Erst 03_features.py und 04_feature_relations.py ausführen."
        )

    train_fe = pd.read_csv(train_pruned_path)

    if TARGET_COL not in train_fe.columns:
        raise KeyError(f"Zielvariable '{TARGET_COL}' nicht in TRAIN_FEATURES gefunden")

    num_features = (
        train_fe.select_dtypes(include=["int64", "float64", "float32"])
        .columns.tolist()
    )
    num_features = [c for c in num_features if c not in [ID_COL, TARGET_COL]]

    # numerische Features als float64 casten, um Probleme mit fehlenden Werten zu vermeiden
    train_fe[num_features] = train_fe[num_features].astype("float64")

    cat_features = (
        train_fe.select_dtypes(include=["category", "object"])
        .columns.tolist()
    )

    for col in cat_features:
        train_fe[col] = train_fe[col].astype("category")

    feature_cols = num_features + cat_features
    X = train_fe[feature_cols]
    y = train_fe[TARGET_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return (
        project_dir,
        feature_cols,
        num_features,
        cat_features,
        X_train,
        X_val,
        y_train,
        y_val,
    )

