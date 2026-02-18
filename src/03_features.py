# src/03_features.py

import pandas as pd
from pathlib import Path
from scipy.stats import f_oneway  # für ANOVA


def get_project_root() -> Path:
    # Diese Datei liegt in src/, Projekt-Root ist eine Ebene darüber.
    return Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    project_dir = get_project_root()

    data_processed_dir = project_dir / "data" / "processed"
    data_features_dir = project_dir / "data" / "features"
    data_features_dir.mkdir(parents=True, exist_ok=True)

    train_clean_path = data_processed_dir / "train_clean.csv"
    test_clean_path = data_processed_dir / "test_clean.csv"

    if not train_clean_path.exists():
        raise FileNotFoundError(f"train_clean.csv nicht gefunden unter: {train_clean_path}")
    if not test_clean_path.exists():
        raise FileNotFoundError(f"test_clean.csv nicht gefunden unter: {test_clean_path}")

    # 1) Daten laden
    train = pd.read_csv(train_clean_path)
    test = pd.read_csv(test_clean_path)

    # 2) Zielvariable & ID-Spalten definieren
    target_col = "productivity_score"    # ggf. anpassen
    main_id_col = "id"                   # bleibt erhalten (falls vorhanden)
    dup_id_col = "student_id"            # soll gedroppt werden

    if target_col not in train.columns:
        raise KeyError(f"Zielvariable '{target_col}' nicht in TRAIN gefunden")

    # 2a) Duplizierte ID-Spalte entfernen, falls vorhanden
    for df_name, df in [("TRAIN", train), ("TEST", test)]:
        if dup_id_col in df.columns:
            print(f"{df_name}: droppe duplizierte ID-Spalte '{dup_id_col}'")
            df.drop(columns=[dup_id_col], inplace=True)

    # 2b) Feature Engineering (direkt auf train und test, identische Schritte)

    # 1) Screentime: Summe aus Medien-/Gaming-Stunden (ohne phone_usage_hours)
    screen_cols = ["social_media_hours", "youtube_hours", "gaming_hours"]
    for df_name, df in [("TRAIN", train), ("TEST", test)]:
        if all(c in df.columns for c in screen_cols):
            df["screentime_hours"] = df[screen_cols].sum(axis=1)
            print(f"{df_name}: Feature 'screentime_hours' erstellt.")

    # 1b) Screentime-to-Sleep-Ratio
    ratio_cols = ["social_media_hours", "youtube_hours", "gaming_hours", "sleep_hours"]
    for df_name, df in [("TRAIN", train), ("TEST", test)]:
        if all(c in df.columns for c in ratio_cols):
            numer = df["social_media_hours"] + df["youtube_hours"] + df["gaming_hours"]
            df["screentime_to_sleep_ratio"] = numer / (df["sleep_hours"] + 1e-3)
            print(f"{df_name}: Feature 'screentime_to_sleep_ratio' erstellt.")

    # 2) Stress-Level in Bins (5 Quantil-Bins)
    for df_name, df in [("TRAIN", train), ("TEST", test)]:
        if "stress_level" in df.columns:
            df["stress_level_bin"] = pd.qcut(
                df["stress_level"], q=5, labels=False, duplicates="drop"
            )
            print(f"{df_name}: Feature 'stress_level_bin' erstellt (5 Quantil-Bins).")

    # 3) Diligence-Score aus assignments, attendance, focus
    diligence_cols = ["assignments_completed", "attendance_percentage", "focus_score"]
    for df_name, df in [("TRAIN", train), ("TEST", test)]:
        if all(c in df.columns for c in diligence_cols):
            sub = df[diligence_cols].astype(float)
            z = (sub - sub.mean()) / sub.std(ddof=0)
            df["diligence_score"] = z.mean(axis=1)
            print(f"{df_name}: Feature 'diligence_score' erstellt (z-standardisiert).")

    # 4) Breaks pro Tag: Bins
    for df_name, df in [("TRAIN", train), ("TEST", test)]:
        if "breaks_per_day" in df.columns:
            df["breaks_per_day_bin"] = pd.cut(
                df["breaks_per_day"],
                bins=[-1, 0, 2, 5, 10, df["breaks_per_day"].max()],
                labels=["0", "1-2", "3-5", "6-10", "11+"]
            )
            print(f"{df_name}: Feature 'breaks_per_day_bin' erstellt.")

    # 5) Productivity-Push-Score aus coffee_intake_mg und exercise_minutes
    push_cols = ["coffee_intake_mg", "exercise_minutes"]
    for df_name, df in [("TRAIN", train), ("TEST", test)]:
        if all(c in df.columns for c in push_cols):
            sub = df[push_cols].astype(float)
            z = (sub - sub.mean()) / sub.std(ddof=0)
            df["productivity_push_score"] = z.mean(axis=1)
            print(f"{df_name}: Feature 'productivity_push_score' erstellt (z-standardisiert).")

    # 6) Recreation-Effort aus sleep_hours und exercise_minutes
    recreation_cols = ["sleep_hours", "exercise_minutes"]
    for df_name, df in [("TRAIN", train), ("TEST", test)]:
        if all(c in df.columns for c in recreation_cols):
            df["recreation_effort_minutes"] = df["sleep_hours"] * 60 + df["exercise_minutes"]
            sleep_bins = pd.qcut(df["sleep_hours"], q=5, labels=False, duplicates="drop")
            exercise_bins = pd.qcut(df["exercise_minutes"], q=5, labels=False, duplicates="drop")
            df["recreation_effort_score"] = sleep_bins + exercise_bins
            print(f"{df_name}: Features 'recreation_effort_minutes' und 'recreation_effort_score' erstellt.")

    # 7) Age-Bins (gleichbreit) + optional age_binned (Quantile)
    for df_name, df in [("TRAIN", train), ("TEST", test)]:
        if "age" in df.columns:
            df["age_bin"] = pd.cut(
                df["age"],
                bins=[0, 13, 15, 17, 20, df["age"].max()],
                labels=["<=13", "14-15", "16-17", "18-20", "21+"]
            )
            df["age_binned"] = pd.qcut(df["age"], q=5, labels=False, duplicates="drop")
            print(f"{df_name}: Features 'age_bin' und 'age_binned' erstellt.")

    # 8) Study-Efficiency
    for df_name, df in [("TRAIN", train), ("TEST", test)]:
        if {"study_hours_per_day", "phone_usage_hours"}.issubset(df.columns):
            df["study_efficiency"] = df["study_hours_per_day"] / (df["phone_usage_hours"] + 1.0)
            print(f"{df_name}: Feature 'study_efficiency' erstellt.")

    # 9) Sleep-Quality
    for df_name, df in [("TRAIN", train), ("TEST", test)]:
        if {"sleep_hours", "focus_score"}.issubset(df.columns):
            df["sleep_quality"] = df["sleep_hours"] * df["focus_score"]
            print(f"{df_name}: Feature 'sleep_quality' erstellt.")

    # 10) Productivity-Index
    for df_name, df in [("TRAIN", train), ("TEST", test)]:
        if {"study_hours_per_day", "focus_score", "attendance_percentage"}.issubset(df.columns):
            df["productivity_index"] = (
                df["study_hours_per_day"]
                * df["focus_score"]
                * df["attendance_percentage"] / 100.0
            )
            print(f"{df_name}: Feature 'productivity_index' erstellt.")

    # 11) Weitere Interaktionen
    for df_name, df in [("TRAIN", train), ("TEST", test)]:
        if {"study_hours_per_day", "focus_score"}.issubset(df.columns):
            df["study_focus_interaction"] = df["study_hours_per_day"] * df["focus_score"]

        if {"sleep_hours", "stress_level"}.issubset(df.columns):
            df["sleep_stress_interaction"] = df["sleep_hours"] * df["stress_level"]

        if {"screentime_hours", "focus_score"}.issubset(df.columns):
            df["screentime_focus_interaction"] = df["screentime_hours"] * df["focus_score"]

        if {"screentime_hours", "diligence_score"}.issubset(df.columns):
            df["screentime_diligence_interaction"] = df["screentime_hours"] * df["diligence_score"]

        if {"attendance_percentage", "focus_score"}.issubset(df.columns):
            df["attendance_focus_interaction"] = df["attendance_percentage"] * df["focus_score"]

        if {"breaks_per_day", "study_efficiency"}.issubset(df.columns):
            df["breaks_study_efficiency"] = df["breaks_per_day"] * df["study_efficiency"]

        if {"age", "screentime_hours"}.issubset(df.columns):
            df["age_screentime_interaction"] = df["age"] * df["screentime_hours"]

    # 12) ANOVA für binned Features
    binned_cols = [
        "stress_level_bin",
        "breaks_per_day_bin",
        "age_bin",
        "age_binned",
    ]

    print("\n===== ANOVA für binned Features =====")
    significant_binned = []

    for col in binned_cols:
        if col not in train.columns:
            continue

        groups = []
        for _, sub_df in train.groupby(col):
            vals = sub_df[target_col].dropna().values
            if len(vals) >= 2:
                groups.append(vals)

        if len(groups) >= 2:
            f_stat, p_value = f_oneway(*groups)
            print(f"{col}: F = {f_stat:.4f}, p = {p_value:.4e}")
            if p_value < 0.05:
                significant_binned.append(col)
        else:
            print(f"{col}: zu wenige Beobachtungen für ANOVA.")

    # 13) numerische Spalten bestimmen (inkl. neuer Features, ohne Ziel)
    num_cols = train.select_dtypes(include=["int64", "float64", "float32"]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    print("\n===== Numerische Spalten (inkl. ID) =====")
    print(num_cols)

    # 14) Korrelation mit Ziel berechnen und nach Schwelle filtern
    corr_with_target = train[num_cols + [target_col]].corr()[target_col].drop(index=target_col)
    print("\n===== Korrelation mit Zielvariable =====")
    print(corr_with_target.sort_values(ascending=False))

    threshold = 0.2
    selected_numeric = corr_with_target[abs(corr_with_target) > threshold].index.tolist()
    selected_numeric = [c for c in selected_numeric if c != main_id_col]

    print(f"\n===== Ausgewählte numerische Features (|corr| > {threshold}) =====")
    print(selected_numeric)

    # Binned-Features, die signifikant sind, zusätzlich aufnehmen
    for col in significant_binned:
        if col in train.columns and col not in selected_numeric:
            selected_numeric.append(col)

    # 15) Features anwenden: gleicher Spalten-Subset auf Train & Test

    # numerische + signifikante binned Features
    all_feature_cols = selected_numeric.copy()
    for col in significant_binned:
        if col in train.columns and col not in all_feature_cols:
            all_feature_cols.append(col)

    print("\n===== Finale Feature-Spalten (numeric + binned) =====")
    print(all_feature_cols)

    keep_train_cols = []
    if main_id_col in train.columns:
        keep_train_cols.append(main_id_col)
    keep_train_cols.extend(all_feature_cols)
    keep_train_cols.append(target_col)

    keep_test_cols = []
    if main_id_col in test.columns:
        keep_test_cols.append(main_id_col)
    keep_test_cols.extend(all_feature_cols)

    train_fe = train[keep_train_cols].copy()
    test_fe = test[keep_test_cols].copy()

    print("\n===== TRAIN Features: Spalten =====")
    print(train_fe.columns.tolist())

    print("\n===== TEST Features: Spalten =====")
    print(test_fe.columns.tolist())

    # 16) Speichern
    train_features_path = data_features_dir / "train_features.csv"
    test_features_path = data_features_dir / "test_features.csv"

    train_fe.to_csv(train_features_path, index=False)
    test_fe.to_csv(test_features_path, index=False)

    print(f"\nFeature-Trainingsdaten gespeichert unter: {train_features_path}")
    print(f"Feature-Testdaten gespeichert unter:      {test_features_path}")
