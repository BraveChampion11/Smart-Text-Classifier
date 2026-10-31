"""Load the saved model and classify text.

Use as a module:
    from predict import predict
    predict("Where is my refund?")   # -> {"label": ..., "confidence": ..., "probabilities": {...}}

Or from the command line:
    python src/predict.py "I was charged twice for the same order"
"""
import sys
from functools import lru_cache
from pathlib import Path

import joblib

from preprocess import clean_text

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "classifier.joblib"


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run `python src/train.py` first."
        )
    return joblib.load(MODEL_PATH)


def predict(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please provide non-empty text.")
    bundle = load_model()
    pipe = bundle["pipeline"]
    probs = pipe.predict_proba([clean_text(text)])[0]
    by_label = {c: float(p) for c, p in zip(pipe.classes_, probs)}
    label = max(by_label, key=by_label.get)
    return {"label": label, "confidence": by_label[label], "probabilities": by_label}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/predict.py "your text here"')
        sys.exit(1)
    result = predict(" ".join(sys.argv[1:]))
    print(f"Predicted: {result['label']} ({result['confidence']:.1%})")
    for lab, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {lab:<10} {p:.1%}")
