"""Run the saved model on the unseen challenge sentences and save the results.

Output: reports/demo_predictions.csv (text, true label, predicted label, confidence, correct)

Run:  python src/demo_predictions.py
"""
from pathlib import Path

import pandas as pd

from predict import predict

ROOT = Path(__file__).resolve().parent.parent

ch = pd.read_csv(ROOT / "data" / "challenge_set.csv", encoding="utf-8")
rows = []
for text, true_label in zip(ch["text"], ch["label"]):
    out = predict(text)
    rows.append(
        {
            "text": text,
            "true_label": true_label,
            "predicted": out["label"],
            "confidence": round(out["confidence"], 3),
            "correct": out["label"] == true_label,
        }
    )

df = pd.DataFrame(rows)
df.to_csv(ROOT / "reports" / "demo_predictions.csv", index=False, encoding="utf-8")
pd.set_option("display.width", 200, "display.max_colwidth", 70)
print(df.to_string(index=False))
print(f"\n{df['correct'].sum()}/{len(df)} correct (final model, trained on all seed data)")
