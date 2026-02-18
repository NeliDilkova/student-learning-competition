# src/01_load_data.py

import pandas as pd
from pathlib import Path


def get_project_root() -> Path:
    """
    Ermittelt das Projekt-Root-Verzeichnis.
    Annahme: Diese Datei liegt in src/, Projekt-Root ist eine Ebene darüber.
    """
    return Path(__file__).resolve().parents[1]


def main() -> None:
    project_dir = get_project_root()

    data_raw_dir = project_dir / "data" / "raw"
    data_raw_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_raw_dir / "train.csv"
    test_path = data_raw_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"train.csv nicht gefunden unter: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"test.csv nicht gefunden unter: {test_path}")

    print(f"Projektverzeichnis: {project_dir}")
    print(f"Lese Trainingsdaten von: {train_path}")
    print(f"Lese Testdaten von:       {test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print("\n=== Train shape ===")
    print(train.shape)

    print("\n=== Test shape ===")
    print(test.shape)

    print("\n=== Train dtypes ===")
    print(train.dtypes)

    print("\n=== Train head ===")
    print(train.head())

    print("\n=== Train describe (numerische Spalten) ===")
    print(train.describe())

    # Optional: Daten zurückgeben, falls später als Modul importiert
    return train, test


if __name__ == "__main__":
    main()
