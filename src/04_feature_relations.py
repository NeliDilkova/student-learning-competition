# src/04_feature_relations.py

import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # nur der Vollständigkeit halber, Plots sind aktuell aus
import matplotlib.pyplot as plt
import seaborn as sns
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

    # stress_level_bin explizit als Kategorie casten (falls vorhanden)
    if "stress_level_bin" in train_fe.columns:
        train_fe["stress_level_bin"] = train_fe["stress_level_bin"].astype("category")
        test_fe["stress_level_bin"] = test_fe["stress_level_bin"].astype("category")

    # Benchmark-Kernfeatures (Roh + engineered aus Benchmark-Modell)
    benchmark_core_features = [
        "age",
        "gender",
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
        "study_efficiency",
        "sleep_quality",
        "distraction_score",
        "productivity_index",
    ]
    benchmark_core_features = [
        f for f in benchmark_core_features if f in train_fe.columns
    ]
    print("\n===== Benchmark-Kernfeatures (immer behalten) =====")
    print(benchmark_core_features)

    # 1) 'id' explizit entfernen
    feature_cols = [c for c in train_fe.columns if c not in [main_id_col, target_col]]

    # 2) nur numerische Features
    num_cols = train_fe[feature_cols].select_dtypes(
        include=["int64", "float64", "float32"]
    ).columns.tolist()

    # kategoriale Features NICHT als numerisch behandeln
    for cat in cat_features_final:
        if cat in num_cols:
            num_cols.remove(cat)

    print("\nNumerische Feature-Spalten (ohne id, ohne Ziel):")
    print(num_cols)

    train_fe.info()

    # 3) Korrelationsmatrix (nur numerisch, ohne Visualisierung)
    corr_matrix = train_fe[num_cols].corr()
    print("\nShape der Korrelationsmatrix:", corr_matrix.shape)
    print(corr_matrix.head())

    # 4) Korrelationen mit Ziel (nur zu Info)
    corr_with_target = train_fe[num_cols + [target_col]].corr()[target_col].drop(
        index=target_col
    )
    print("\nKorrelationen mit Ziel:")
    print(corr_with_target.sort_values(ascending=False))

    # ===== NEUE LOGIK: Benchmark-Anker + schwach korrelierte Zusatzfeatures =====
    # Korrelationen aller numerischen Features (inkl. Benchmark-Features)
    corr_all = train_fe[num_cols + [target_col]].corr()

    # Benchmark-Features als Anker im finalen Set
    final_num_cols = [f for f in benchmark_core_features if f in num_cols]

    # alle anderen numerischen Features (nicht Benchmark)
    other_num_features = [f for f in num_cols if f not in benchmark_core_features]

    max_corr_threshold = 0.6

    for f in other_num_features:
        # Wenn dieses Feature NaNs in Korrelation hat, überspringen
        if f not in corr_all.index:
            continue

        # Korrelation dieses Features zu allen Benchmark-Features, die in corr_all vorhanden sind
        valid_anchors = [bf for bf in final_num_cols if bf in corr_all.columns]
        if not valid_anchors:
            # falls aus irgendeinem Grund keine Anker in der Matrix sind, alles zulassen
            final_num_cols.append(f)
            continue

        corrs_to_benchmark = corr_all.loc[f, valid_anchors].abs()
        max_corr_to_benchmark = corrs_to_benchmark.max()

        if max_corr_to_benchmark < max_corr_threshold:
            final_num_cols.append(f)

    print("\n===== Finale numerische Features nach Benchmark-Logik (ohne log) =====")
    print(final_num_cols)

    # 6) Log-Transformation prüfen (Train + Test identisch erweitern)
    log_suffix = "_log"
    log_candidates = final_num_cols

    corr_with_target_final = (
        train_fe[final_num_cols + [target_col]]
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

        corr_orig = abs(corr_with_target_final[col])

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
    final_num_cols_with_log = []
    for col in final_num_cols:
        if col in better_log_features:
            final_num_cols_with_log.append(log_feature_names[col])
        else:
            final_num_cols_with_log.append(col)

    print("\n===== Finale numerische Features (inkl. log-Entscheidung) =====")
    print(final_num_cols_with_log)

    # 8) Neues, bereinigtes Feature-Set speichern (id + Ziel + numerische + kategoriale)
    keep_train_cols = []
    if main_id_col in train_fe.columns:
        keep_train_cols.append(main_id_col)
    keep_train_cols.extend(final_num_cols_with_log)
    keep_train_cols.extend(cat_features_final)
    if target_col in train_fe.columns:
        keep_train_cols.append(target_col)

    keep_test_cols = []
    if main_id_col in test_fe.columns:
        keep_test_cols.append(main_id_col)
    keep_test_cols.extend(final_num_cols_with_log)
    keep_test_cols.extend(cat_features_final)

    train_fe_pruned = train_fe[keep_train_cols].copy()
    test_fe_pruned = test_fe[keep_test_cols].copy()

    train_pruned_path = data_features_dir / "train_features_pruned.csv"
    test_pruned_path = data_features_dir / "test_features_pruned.csv"

    train_fe_pruned.to_csv(train_pruned_path, index=False)
    test_fe_pruned.to_csv(test_pruned_path, index=False)

    print("\nDEBUG: Komme am Ende des Skripts an, vor dem Speichern.")
    print(f"\nBereinigte Feature-Trainingsdaten gespeichert unter: {train_pruned_path}")
    print(f"Bereinigte Feature-Testdaten gespeichert unter:      {test_pruned_path}")
