# Smart Text Classifier

A small NLP application that classifies short text into one of four categories:
**Complaint**, **Inquiry**, **Feedback** or **Other**. Built for the Varolline
Internship Program (AI Engineer technical assignment).

- Model: TF-IDF (word 1-2 grams) + Logistic Regression, compared against TF-IDF + Multinomial Naive Bayes
- Interface: Streamlit web app and a command-line predictor
- Evaluation: accuracy, macro-F1, weighted-F1, per-class report, confusion matrix, grouped cross-validation, plus an unseen "challenge set"

## Results

| Evaluation | Accuracy | Macro-F1 |
|---|---|---|
| 5-fold grouped CV, Naive Bayes (train part) | 0.664 | 0.659 |
| 5-fold grouped CV, Logistic Regression (train part) | 0.765 | 0.762 |
| **Held-out test set (120 rows), Logistic Regression** | **0.867** | **0.865** |
| Held-out test set, Naive Bayes | 0.792 | 0.787 |
| Unseen challenge set (32 new sentences), Logistic Regression | 0.813 | 0.809 |

Logistic Regression was selected because it had the higher cross-validated macro-F1.
Full numbers: `reports/metrics.json`, `reports/classification_report.txt`,
`reports/confusion_matrix.png`.

## Setup (Windows / PowerShell)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux use `source venv/bin/activate`.

## How to run

Run all commands from the project root.

```powershell
python src\build_dataset.py      # (optional) regenerate data\dataset.csv and data\challenge_set.csv
python src\check_dataset.py      # dataset sanity checks
python src\train.py              # train, compare models, evaluate, save models\classifier.joblib
python src\demo_predictions.py   # predictions for 32 unseen sentences -> reports\demo_predictions.csv
python tests_smoke.py            # quick tests
streamlit run src\app.py         # web interface (opens in the browser)
python src\predict.py "I was charged twice for the same order"   # command line
```

Re-run `python src\train.py` once after installing so the saved model matches your library versions.

## Features

- Cleaning that keeps useful signals: `?` and `!` become tokens, URLs and numbers become placeholders, stopwords are kept (words like *not*, *how*, *why* decide the label)
- Leakage-safe evaluation: variants of the same seed sentence are grouped and never split across train and test
- Class probabilities and a low-confidence notice in the web interface
- Challenge set that is never used for training or model selection

## Dataset

`data/dataset.csv` (588 rows: Complaint 153, Other 150, Feedback 144, Inquiry 141) with columns `text`, `label`, `group`.

**How it was created.** No public dataset was used. I wrote 196 seed sentences by hand
(`src/seed_data.py`) following the label definitions and tie-breaker rules in
`data/LABELS.md`, across five domains (food delivery, banking app, online shop, university
portal, telecom) plus domain-free sentences. The seeds were drafted with help from an AI
assistant (Claude) and then reviewed and edited by me. `src/build_dataset.py` adds up to
two light variations per seed (greeting/sign-off, lower-casing, dropped full stop, one
typo). The same variations are used for every class, so they carry no label information.
`group` identifies the seed, so evaluation can keep variants together.

Hard cases are included deliberately: inquiries without a question mark, polite complaints,
complaints phrased as questions, feedback containing negative words, and "Other" text that
looks like an inquiry ("What is the meaning of life?").

`data/challenge_set.csv` holds 32 further hand-written sentences with different wording,
used only for the final unseen-text check.

## Sample predictions (12 of 32, final model)

| Text | True | Predicted | Confidence |
|---|---|---|---|
| The courier left my package outside in the rain and now it is ruined. | Complaint | Complaint | 0.61 |
| I have been on hold for 45 minutes and still no one has answered. | Complaint | Complaint | 0.94 |
| I would be thankful if someone could look into why my account was suspended. | Complaint | Complaint | 0.91 |
| need invoice copy for march | Inquiry | Inquiry | 0.94 |
| Where can I track my refund? | Inquiry | Inquiry | 0.81 |
| Is it possible to switch to the yearly plan? | Inquiry | **Feedback** | 0.52 |
| The new update looks great, though the font is a little small. | Feedback | Feedback | 0.88 |
| Support agent was kind and explained everything patiently. | Feedback | **Complaint** | 0.96 |
| Maybe add a search bar on the home page? | Feedback | Feedback | 0.46 |
| Good afternoon everyone | Other | Other | 0.83 |
| You have been selected for a free vacation, call now | Other | Other | 0.50 |
| What is the tallest mountain in the world? | Other | **Inquiry** | 0.89 |

All 32 are in `reports/demo_predictions.csv` (27 of 32 correct).

## Limitations

- **Small, partly synthetic data.** Many rows are variations of the same seeds, so the model sees limited vocabulary. Scores on real customer messages will be lower.
- **Style shortcuts.** Complaints are longer and "Other" texts are shorter in this dataset; the top Complaint features include function words such as "and", "was", "my". A polite, positive message that mentions an agent ("Support agent was kind...") can be mislabelled.
- **Off-topic questions** look like inquiries ("What is the tallest mountain?") and are often misclassified; there is no out-of-scope detector.
- **Mixed intent.** Each text gets one label, but real messages often combine a complaint and a question.
- **English only**, no sarcasm handling, and probabilities are not calibrated.

## Future improvements

1. Fine-tune a small transformer (for example DistilBERT) or use sentence embeddings with Logistic Regression, which handle meaning better than word counts.
2. Collect and label real messages (with permission), including Roman Urdu / Urdu, and add a confidence threshold that routes low-confidence texts to "Other" or to a human.
3. Allow multi-label output so a message can be both Complaint and Inquiry.
4. Calibrate probabilities and add an explicit out-of-scope class trained on more off-topic text.

## Project structure

```
data/        dataset.csv, challenge_set.csv, LABELS.md
src/         seed_data.py, build_dataset.py, check_dataset.py, preprocess.py,
             train.py, predict.py, demo_predictions.py, app.py
models/      classifier.joblib (saved pipeline)
reports/     metrics.json, classification_report.txt, confusion_matrix.png,
             demo_predictions.csv, top_features.txt, REPORT.md / REPORT.pdf
tests_smoke.py
```

## Acknowledgements

Built with scikit-learn, pandas, matplotlib and Streamlit. An AI assistant (Claude) was
used to help draft code and seed sentences; I reviewed and can explain all of it.
