"""Sanity checks for data/dataset.csv: balance, duplicates, length and '?' shortcuts.

Run:  python src/check_dataset.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "dataset.csv", encoding="utf-8")

print("Rows:", len(df), "| seed groups:", df["group"].nunique())
print("\nClass counts:")
print(df["label"].value_counts().to_string())

print("\nMissing values:")
print(df.isna().sum().to_string())

dupes = df[df.duplicated(subset="text", keep=False)]
print(f"\nDuplicate texts: {len(dupes)}")
if len(dupes):
    print(dupes.sort_values("text").to_string())

df["n_words"] = df["text"].str.split().str.len()
print("\nWords per text by class:")
print(df.groupby("label")["n_words"].describe()[["mean", "min", "max"]].round(1).to_string())

df["has_qmark"] = df["text"].str.contains(r"\?")
print("\nShare of texts containing '?' by class:")
print(df.groupby("label")["has_qmark"].mean().round(2).to_string())
