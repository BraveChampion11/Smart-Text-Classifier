"""Train, compare and evaluate the text classifier.

Steps
  1. Load data/dataset.csv and clean the text.
  2. Group-aware stratified 80/20 train/test split (variants of one seed
     sentence never end up on both sides -> no leakage).
  3. Compare TF-IDF + Naive Bayes vs TF-IDF + Logistic Regression using
     5-fold grouped cross-validation on the TRAINING part only.
  4. Report accuracy, macro-F1, weighted-F1, per-class report and confusion
     matrix on the held-out test set.
  5. Evaluate on data/challenge_set.csv (never used for training/selection).
  6. Refit the chosen model on ALL data and save it to models/classifier.joblib.

Run:  python src/train.py
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from preprocess import clean_text

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["Complaint", "Inquiry", "Feedback", "Other"]
RANDOM_STATE = 42


def make_pipeline(classifier) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("clf", classifier),
        ]
    )


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
    }


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "dataset.csv", encoding="utf-8")
    df["clean"] = df["text"].map(clean_text)
    X, y, groups = df["clean"], df["label"], df["group"]

    # 80/20 split: take one fold of a 5-fold stratified, grouped split.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    g_train = groups.iloc[train_idx]
    assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[test_idx]))
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    candidates = {
        "naive_bayes": make_pipeline(MultinomialNB(alpha=0.5)),
        "logistic_regression": make_pipeline(
            LogisticRegression(C=10, max_iter=2000, random_state=RANDOM_STATE)
        ),
    }

    # --- model comparison via grouped cross-validation on the training part ---
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, pipe in candidates.items():
        scores = cross_validate(
            pipe, X_train, y_train, groups=g_train, cv=cv,
            scoring=["accuracy", "f1_macro"],
        )
        results[name] = {
            "cv_accuracy_mean": round(scores["test_accuracy"].mean(), 4),
            "cv_accuracy_std": round(scores["test_accuracy"].std(), 4),
            "cv_macro_f1_mean": round(scores["test_f1_macro"].mean(), 4),
            "cv_macro_f1_std": round(scores["test_f1_macro"].std(), 4),
        }
        print(f"[CV] {name}: acc={results[name]['cv_accuracy_mean']:.3f} "
              f"+/- {results[name]['cv_accuracy_std']:.3f}, "
              f"macro-F1={results[name]['cv_macro_f1_mean']:.3f}")

    best_name = max(results, key=lambda n: results[n]["cv_macro_f1_mean"])
    print(f"\nSelected model (best CV macro-F1): {best_name}")

    # --- held-out test evaluation for every candidate ---
    fitted = {}
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        results[name]["test"] = metrics(y_test, pipe.predict(X_test))
        print(f"[TEST] {name}: {results[name]['test']}")

    best = fitted[best_name]
    y_pred = best.predict(X_test)
    report_text = classification_report(y_test, y_pred, labels=LABELS, digits=3)
    print("\nClassification report (held-out test set, selected model):\n" + report_text)

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix (test set) - {best_name}")
    plt.tight_layout()
    fig.savefig(reports_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # --- challenge set: unseen, differently-worded sentences ---
    ch = pd.read_csv(ROOT / "data" / "challenge_set.csv", encoding="utf-8")
    ch_pred = best.predict(ch["text"].map(clean_text))
    results["challenge_set"] = {
        "model": best_name,
        "note": "model trained on the train split only; challenge set never used for fitting or selection",
        **metrics(ch["label"], ch_pred),
    }
    print(f"[CHALLENGE] {results['challenge_set']}")
    ch_cm = confusion_matrix(ch["label"], ch_pred, labels=LABELS)
    print(pd.DataFrame(ch_cm, index=LABELS, columns=LABELS).to_string())

    # --- refit the selected model on ALL data for deployment ---
    final = make_pipeline(
        MultinomialNB(alpha=0.5) if best_name == "naive_bayes"
        else LogisticRegression(C=10, max_iter=2000, random_state=RANDOM_STATE)
    )
    final.fit(X, y)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"pipeline": final, "labels": LABELS, "model_name": best_name},
                ROOT / "models" / "classifier.joblib")

    results["selected_model"] = best_name
    results["split"] = {"train_rows": int(len(X_train)), "test_rows": int(len(X_test))}
    (reports_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # most influential words (logistic regression only) - helps error analysis
    if best_name == "logistic_regression":
        tfidf, clf = final.named_steps["tfidf"], final.named_steps["clf"]
        vocab = np.array(tfidf.get_feature_names_out())
        lines = []
        for i, label in enumerate(clf.classes_):
            top = vocab[np.argsort(clf.coef_[i])[-10:][::-1]]
            lines.append(f"{label}: {', '.join(top)}")
        (reports_dir / "top_features.txt").write_text("\n".join(lines), encoding="utf-8")
        print("\nTop features per class:\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
