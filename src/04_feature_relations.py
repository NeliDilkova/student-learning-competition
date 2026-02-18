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

    # 0) finale kategoriale Features festlegen
    cat_features_final = [
        "stress_level_bin",
        "breaks_per_day_bin",
        "age_bin",
    ]
    cat_features_final = [c for c in cat_features_final if c in train_fe.columns]
    print("\n===== Finale kategoriale Features =====")
    print(cat_features_final)

    # 1) 'id' explizit entfernen
    feature_cols = [c for c in train_fe.columns if c not in [main_id_col, target_col]]

    # 2) nur numerische Features für Korrelationsanalyse
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

    # 5) Hoch korrelierte Feature-Paare filtern (|corr| > 0.6) – konservative Variante
    high_corr_threshold = 0.6
    corr_feat = train_fe[num_cols].corr().abs()

    # nach |corr mit Ziel| sortieren
    abs_target_corr = corr_with_target.abs().reindex(num_cols).fillna(0.0)
    sorted_features = abs_target_corr.sort_values(ascending=False).index.tolist()

    kept_features = []
    dropped_features = set()

    for f in sorted_features:
        if f in dropped_features:
            continue

        kept_features.append(f)

        high_corr_partners = corr_feat.index[corr_feat[f] > high_corr_threshold].tolist()
        if f in high_corr_partners:
            high_corr_partners.remove(f)

        for partner in high_corr_partners:
            dropped_features.add(partner)

    print(
        f"\n===== Stark korrelierte Features (|corr| > {high_corr_threshold}) – gedroppte: ====="
    )
    print(dropped_features)

    pruned_num_cols = kept_features
    print("\n===== Numerische Features nach Hochkorrelations-Filter =====")
    print(pruned_num_cols)

    # 6) Log-Transformation prüfen (Train + Test identisch erweitern)
    log_suffix = "_log"
    log_candidates = pruned_num_cols

    corr_with_target_pruned = (
        train_fe[pruned_num_cols + [target_col]]
        .corr()[target_col]
        .drop(index=target_col)
    )

    better_log_features = []
    log_feature_names = {}

    for col in log_candidates:
        col_values_train = train_fe[col]

        if (col_values_train <= 0).any():
            continue

        log_col = col + log_suffix
        log_feature_names[col] = log_col

        train_fe[log_col] = np.log(col_values_train)
        if col in test_fe.columns:
            test_fe[log_col] = np.log(test_fe[col].clip(lower=1e-6))

        corr_orig = abs(corr_with_target_pruned[col])

        corr_log = abs(
            train_fe[[log_col, target_col]]
            .corr()[target_col]
            .drop(index=target_col)
            .iloc[0]
        )

        print(f"{col}: |corr| = {corr_orig:.3f}, {log_col}: |corr| = {corr_log:.3f}")

        if corr_log > corr_orig:
            better_log_features.append(col)

    print("\n===== Features, bei denen log-Version besser ist =====")
    print(better_log_features)

    # 7) Finale numerische Features: Original vs. log
    final_num_cols = []
    for col in pruned_num_cols:
        if col in better_log_features:
            final_num_cols.append(log_feature_names[col])
        else:
            final_num_cols.append(col)

    print("\n===== Finale numerische Features (inkl. log-Entscheidung) =====")
    print(final_num_cols)

    # 8) Neues, bereinigtes Feature-Set speichern (id + Ziel + numerische + kategoriale)
    keep_train_cols = []
    if main_id_col in train_fe.columns:
        keep_train_cols.append(main_id_col)
    keep_train_cols.extend(final_num_cols)
    keep_train_cols.extend(cat_features_final)
    if target_col in train_fe.columns:
        keep_train_cols.append(target_col)

    keep_test_cols = []
    if main_id_col in test_fe.columns:
        keep_test_cols.append(main_id_col)
    keep_test_cols.extend(final_num_cols)
    keep_test_cols.extend(cat_features_final)

    train_fe_pruned = train_fe[keep_train_cols].copy()
    test_fe_pruned = test_fe[keep_test_cols].copy()

    train_pruned_path = data_features_dir / "train_features_pruned.csv"
    test_pruned_path = data_features_dir / "test_features_pruned.csv"

    train_fe_pruned.to_csv(train_pruned_path, index=False)
    test_fe_pruned.to_csv(test_pruned_path, index=False)

    print(f"\nBereinigte Feature-Trainingsdaten gespeichert unter: {train_pruned_path}")
    print(f"Bereinigte Feature-Testdaten gespeichert unter:      {test_pruned_path}")
