"""Minimal smoke tests. Run from the project root:  python tests_smoke.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocess import clean_text  # noqa: E402
from predict import predict  # noqa: E402

# preprocessing keeps the punctuation signal and drops noise
assert "qmark" in clean_text("Where is my refund?")
assert "urltoken" in clean_text("see http://x.com now")
assert "not" in clean_text("It is NOT working")  # stopwords are kept on purpose

# prediction returns a valid label and probabilities that sum to 1
out = predict("What are your working hours?")
assert out["label"] in {"Complaint", "Inquiry", "Feedback", "Other"}
assert abs(sum(out["probabilities"].values()) - 1) < 1e-6

# empty input is rejected
try:
    predict("   ")
    raise SystemExit("empty text should raise ValueError")
except ValueError:
    pass

# a clear example for every class
cases = {
    "I was charged twice and nobody is replying": "Complaint",
    "How do I reset my password?": "Inquiry",
    "Would be nice to have a dark mode": "Feedback",
    "Good morning": "Other",
}
for text, expected in cases.items():
    got = predict(text)["label"]
    assert got == expected, f"{text!r}: expected {expected}, got {got}"

print("All smoke tests passed.")
