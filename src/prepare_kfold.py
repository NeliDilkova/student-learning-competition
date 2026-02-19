# src/prepare_kfold.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import KFold

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def load_full_data():
    project_dir = get_project_root()
    data_features_dir = project_dir / "data" / "features"

    train_path = data_features_dir / "train_features_pruned.csv"
    train_fe = pd.read_csv(train_path)

    target_col = "productivity_score"
    main_id_col = "id"

    feature_cols = [c for c in train_fe.columns if c not in [main_id_col, target_col]]
    X = train_fe[feature_cols]
    y = train_fe[target_col]

    # falls du kategoriale Features brauchst:
    cat_features = [c for c in X.columns if X[c].dtype.name == "category"]
    num_features = [c for c in X.columns if c not in cat_features]

    return project_dir, feature_cols, num_features, cat_features, X, y

def get_kfold(n_splits=5, random_state=42):
    return KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
