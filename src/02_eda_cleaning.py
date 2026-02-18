# src/02_eda_cleaning.py

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway


def get_project_root() -> Path:
    # Diese Datei liegt in src/, Projekt-Root ist eine Ebene darüber.
    return Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    # 1) Pfade setzen
    project_dir = get_project_root()
    data_raw_dir = project_dir / "data" / "raw"
    data_processed_dir = project_dir / "data" / "processed"
    data_processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_raw_dir / "train.csv"
    test_path = data_raw_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"train.csv nicht gefunden unter: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"test.csv nicht gefunden unter: {test_path}")

    # 2) Daten einlesen
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Zielvariable festlegen
    target_col = "productivity_score"  # ggf. anpassen!

    # Sicherstellen, dass Zielvariable existiert
    if target_col not in train.columns:
        raise KeyError(f"Zielvariable '{target_col}' nicht in TRAIN gefunden")

    # 3) EDA – Basisinformationen TRAIN
    print("\n===== TRAIN: shape =====")
    print(train.shape)

    print("\n===== TRAIN: dtypes =====")
    print(train.dtypes)

    print("\n===== TRAIN: missing values =====")
    print(train.isna().sum())

    print("\n===== TRAIN: describe (numerische Spalten) =====")
    print(train.describe())

    # 4) EDA – Basisinformationen TEST
    print("\n===== TEST: shape =====")
    print(test.shape)

    print("\n===== TEST: dtypes =====")
    print(test.dtypes)

    print("\n===== TEST: missing values =====")
    print(test.isna().sum())

    print("\n===== TEST: describe (numerische Spalten) =====")
    print(test.describe())

    # 5) Numerische Verteilungen (TRAIN)
    num_cols = train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    print("\n===== TRAIN: numerische Spalten =====")
    print(num_cols)

    for col in num_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(train[col], kde=True)
        plt.title(f"Verteilung von {col} (TRAIN)")
        plt.tight_layout()
        plt.show()

    # 6) Kategorische Verteilungen (TRAIN)
    cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
    print("\n===== TRAIN: kategorische Spalten =====")
    print(cat_cols)

    for col in cat_cols:
        plt.figure(figsize=(8, 4))
        value_counts = train[col].value_counts().head(20)
        sns.barplot(x=value_counts.index, y=value_counts.values)
        plt.title(f"Verteilung von {col} (TRAIN) – Top 20 Kategorien")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    # 7) Kategorische Variablen: Zusammenhang mit Zielvariable (deskriptiv + ANOVA)
    for col in cat_cols:
        print(f"\n===== Kategorie vs. Ziel: {col} =====")

        # a) Deskriptive Statistik pro Kategorie
        grouped = (
            train.groupby(col)[target_col]
            .agg(["count", "mean", "std"])
            .sort_values("mean", ascending=False)
        )
        print("\n--- Deskriptive Statistik ---")
        print(grouped.head(20))

        # b) ANOVA: Mittelwertvergleich über Kategorien
        groups = []
        for _, sub_df in train.groupby(col):
            vals = sub_df[target_col].dropna().values
            if len(vals) >= 2:
                groups.append(vals)

        if len(groups) >= 2:
            f_stat, p_value = f_oneway(*groups)
            print("\n--- ANOVA ---")
            print(f"F-Statistik: {f_stat:.4f}, p-Wert: {p_value:.4e}")
        else:
            print("\n--- ANOVA ---")
            print("Zu wenige Kategorien/Beobachtungen für einen sinnvollen ANOVA-Test.")

        # c) Boxplot Zielverteilung je Kategorie (optisch)
        if train[col].nunique() <= 15:
            plt.figure(figsize=(8, 4))
            sns.boxplot(data=train, x=col, y=target_col)
            plt.title(f"{target_col} nach {col}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()
        else:
            top_k = 10
            top_categories = train[col].value_counts().head(top_k).index
            subset = train[train[col].isin(top_categories)]

            plt.figure(figsize=(8, 4))
            sns.boxplot(data=subset, x=col, y=target_col)
            plt.title(f"{target_col} nach {col} (Top {top_k} Kategorien)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()

    # 8) Zusammenhang numerische Features mit Zielvariable
    corr_with_target = train[num_cols].corr()[target_col].sort_values(ascending=False)
    print("\n===== Korrelation numerischer Features mit Ziel =====")
    print(corr_with_target)

    plt.figure(figsize=(6, 8))
    sns.barplot(x=corr_with_target.values, y=corr_with_target.index)
    plt.title(f"Korrelation mit Ziel '{target_col}'")
    plt.tight_layout()
    plt.show()

    # 9) Korrelationen zwischen numerischen Variablen (Heatmap)
    plt.figure(figsize=(10, 8))
    corr_matrix = train[num_cols].corr()
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0)
    plt.title("Korrelationsmatrix numerischer Features (TRAIN)")
    plt.tight_layout()
    plt.show()

    # 10) Einfaches Cleaning – Duplikate
    print("\n===== Entferne Duplikate in TRAIN =====")
    before = train.shape[0]
    train = train.drop_duplicates()
    after = train.shape[0]
    print(f"TRAIN: {before - after} Duplikate entfernt, neue Zeilenanzahl: {after}")

    print("\n===== Entferne Duplikate in TEST =====")
    before_test = test.shape[0]
    test = test.drop_duplicates()
    after_test = test.shape[0]
    print(f"TEST: {before_test - after_test} Duplikate entfernt, neue Zeilenanzahl: {after_test}")

    # TODO: Missing-Value-Handling und weitere Cleaning-Schritte ergänzen

    # 11) Speichern
    train_clean_path = data_processed_dir / "train_clean.csv"
    test_clean_path = data_processed_dir / "test_clean.csv"

    train.to_csv(train_clean_path, index=False)
    test.to_csv(test_clean_path, index=False)

    print(f"\nBereinigte Trainingsdaten gespeichert unter: {train_clean_path}")
    print(f"Bereinigte Testdaten gespeichert unter:      {test_clean_path}")
