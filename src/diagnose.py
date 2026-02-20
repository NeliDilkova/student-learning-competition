import pandas as pd
sub = pd.read_csv("models/submission_v2_benchmark.csv")
print(sub["productivity_score"].describe())
print("\nErste 10 Zeilen:")
print(sub.head(10))
print("\nAnzahl Zeilen:", len(sub))

from pathlib import Path

# Finde sample_submission
for subdir in ["raw", "processed", "data"]:
    p = Path("data") / subdir / "sample_submission.csv"
    if p.exists():
        sample = pd.read_csv(p)
        print(f"\nSample Submission ({p}):")
        print(sample.shape)
        print(sample.head())
        break


RAW = Path(r"C:\Users\nelid\Documents\Kaggle Competitions\Student Learning Competition\student-learning-competition\data\raw")

print("Dateien in data/raw/:")
for f in sorted(RAW.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size / 1024:.1f} KB)")

import pandas as pd

sample = pd.read_csv(RAW / "sample_submission.csv")
print("\nsample_submission.csv:")
print(sample.head())
print("productivity_score Werte:", sample["productivity_score"].unique()[:5])
print("Sind alle Nullen?", (sample["productivity_score"] == 0).all())

test = pd.read_csv(RAW / "test.csv")
print("\ntest.csv Spalten:", test.columns.tolist())
print("Hat productivity_score:", "productivity_score" in test.columns)