# src/04_feature_relations.py

import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    project_dir = get_project_root()
    data_features_dir = project_dir / "data" / "features"

    train_features_path = data_features_dir / "train_features.csv"
    test_features_path = data_features_dir / "test_features.csv"

    if not train_features_path.exists():
        raise FileNotFoundError(
            f"{train_features_path} nicht gefunden. Erst 03_features.py ausführen."
        )
    if not test_features_path.exists():
        raise FileNotFoundError(
            f"{test_features_path} nicht gefunden. Erst 03_features.py ausführen."
        )

    train_fe = pd.read_csv(train_features_path)
    test_fe = pd.read_csv(test_features_path)

    target_col = "productivity_score"
    main_id_col = "id"

    # 1) 'id' explizit entfernen
    feature_cols = [c for c in train_fe.columns if c not in [main_id_col, target_col]]

    # 2) nur numerische Features für Korrelationsheatmap
    num_cols = train_fe[feature_cols].select_dtypes(
        include=["int64", "float64", "float32"]
    ).columns.tolist()

    print("\nNumerische Feature-Spalten (ohne id, ohne Ziel):")
    print(num_cols)

    # 3) Korrelationsmatrix (optional visual)
    corr_matrix = train_fe[num_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
    plt.title("Korrelationsmatrix der erklärenden numerischen Variablen (ohne id)")
    plt.tight_layout()
    plt.show()

    # 4) Korrelationen mit Ziel
    corr_with_target = train_fe[num_cols + [target_col]].corr()[target_col].drop(
        index=target_col
    )
    print("\nKorrelationen mit Ziel:")
    print(corr_with_target.sort_values(ascending=False))

    # Top-5 für Pairplot
    top_features = (
        corr_with_target.abs().sort_values(ascending=False).head(5).index.tolist()
    )
    print("\nTop-5 Features nach |corr| mit Ziel:")
    print(top_features)

    sns.pairplot(
        train_fe[top_features + [target_col]],
        corner=True,
        diag_kind="hist",
    )
    plt.suptitle("Pairplot Top-Features vs. productivity_score", y=1.02)
    plt.show()

    # 5) Hoch korrelierte Feature-Paare filtern (|corr| > 0.6)
    high_corr_threshold = 0.6

    corr_feat = train_fe[num_cols].corr().abs()
    upper = corr_feat.where(
        np.triu(np.ones(corr_feat.shape), k=1).astype(bool)
    )

    to_drop = set()

    for col in upper.columns:
        highly_corr_with_col = upper.index[upper[col] > high_corr_threshold].tolist()
        for other in highly_corr_with_col:
            corr_col = abs(corr_with_target.get(col, 0.0))
            corr_other = abs(corr_with_target.get(other, 0.0))

            if corr_col >= corr_other:
                to_drop.add(other)
            else:
                to_drop.add(col)

    print(
        f"\n===== Stark korrelierte Features (|corr| > {high_corr_threshold}) – zu droppen: ====="
    )
    print(to_drop)

    # 6) Featureliste bereinigen
    pruned_num_cols = [f for f in num_cols if f not in to_drop]
    print("\n===== Numerische Features nach Hochkorrelations-Filter =====")
    print(pruned_num_cols)

    # 7) Neues, bereinigtes Feature-Set speichern
    #    (id und Ziel unverändert, nur numerische Features bereinigt)
    keep_train_cols = []
    if main_id_col in train_fe.columns:
        keep_train_cols.append(main_id_col)
    keep_train_cols.extend(pruned_num_cols)
    if target_col in train_fe.columns:
        keep_train_cols.append(target_col)

    keep_test_cols = []
    if main_id_col in test_fe.columns:
        keep_test_cols.append(main_id_col)
    keep_test_cols.extend(pruned_num_cols)

    train_fe_pruned = train_fe[keep_train_cols].copy()
    test_fe_pruned = test_fe[keep_test_cols].copy()

    train_pruned_path = data_features_dir / "train_features_pruned.csv"
    test_pruned_path = data_features_dir / "test_features_pruned.csv"

    train_fe_pruned.to_csv(train_pruned_path, index=False)
    test_fe_pruned.to_csv(test_pruned_path, index=False)

    print(f"\nBereinigte Feature-Trainingsdaten gespeichert unter: {train_pruned_path}")
    print(f"Bereinigte Feature-Testdaten gespeichert unter:      {test_pruned_path}")
