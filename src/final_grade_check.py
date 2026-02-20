# src/check_final_grade.py
import pandas as pd
from pathlib import Path

RAW = Path(r"C:\Users\nelid\Documents\Kaggle Competitions\Student Learning Competition\student-learning-competition\data\raw")

train = pd.read_csv(RAW / "train.csv")
test  = pd.read_csv(RAW / "test.csv")

print("=== final_grade ===")
print(f"Train: min={train['final_grade'].min():.1f}, max={train['final_grade'].max():.1f}, mean={train['final_grade'].mean():.1f}, std={train['final_grade'].std():.1f}")
print(f"Test:  min={test['final_grade'].min():.1f}, max={test['final_grade'].max():.1f}, mean={test['final_grade'].mean():.1f}, std={test['final_grade'].std():.1f}")

print("\n=== productivity_score vs final_grade Korrelation (Train) ===")
corr = train["productivity_score"].corr(train["final_grade"])
print(f"Korrelation: {corr:.4f}")

print("\n=== Alle Korrelationen mit productivity_score ===")
num_cols = train.select_dtypes(include="number").columns.tolist()
num_cols.remove("productivity_score")
num_cols = [c for c in num_cols if c != "id" and c != "student_id"]
corrs = train[num_cols + ["productivity_score"]].corr()["productivity_score"].drop("productivity_score")
print(corrs.sort_values(ascending=False).to_string())
