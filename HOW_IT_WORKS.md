# How it works: study guide (for your review, not required by the brief)

Read this, run each script once, and change one thing (see the exercises) so you can explain every part.

## The pipeline in one picture

```
seed_data.py ──> build_dataset.py ──> data/dataset.csv ──┐
                                                         ├─> train.py ──> models/classifier.joblib
data/challenge_set.csv (never trained on) ───────────────┘        │
                                                                  ├─> reports/ (metrics, confusion matrix)
predict.py  <── loads the model, cleans text with preprocess.py   │
app.py (Streamlit) and demo_predictions.py both call predict.py <─┘
```

## Concepts you should be able to explain

**TF-IDF.** Each text becomes a vector of word weights. TF = how often a word appears in the text; IDF = how rare it is across all texts, so common words get less weight. `ngram_range=(1,2)` adds two-word phrases such as "not working". `sublinear_tf=True` uses log(1 + count) so repeating a word does not dominate.

**Logistic Regression.** Learns one weight per word per class and turns the weighted sum into probabilities with softmax. `C=10` means weaker regularisation (the model may fit the training data more closely). It is linear, so you can inspect weights (`reports/top_features.txt`).

**Multinomial Naive Bayes.** Assumes words are independent given the class and multiplies word probabilities. Fast and a strong baseline, but it was weaker here because the data is small and word order and phrases matter.

**Why group-aware splitting?** The dataset has 2-3 near-identical variants per seed sentence. A random split would put a twin in train and test, so the model "recognises" the test sentence and the score is inflated. `StratifiedGroupKFold` keeps each seed's variants together while keeping class balance.

**Why a challenge set?** The main data is synthetic and templated, so even a group-aware test set shares style with the training data. The challenge set is new wording and gives a more honest picture (0.81 vs 0.87).

**Accuracy vs macro-F1.** Accuracy is the share of correct predictions. F1 balances precision (of the texts I labelled X, how many were X) and recall (of the real X texts, how many I found). Macro-F1 averages F1 over classes equally, so a weak class cannot hide.

**Why keep stopwords?** "not working" vs "working", "how do I" vs "I do". Removing "not", "how", "why" would destroy the signal.

**Why `?` becomes the token `qmark`?** The default tokenizer drops punctuation, but a question mark is a strong Inquiry hint. The dataset check shows it is not a perfect shortcut: only 70% of Inquiries contain one and 22% of Other texts do.

## Likely review questions and short answers

1. *Why this model?* Small data, need for explainability, fast training; Logistic Regression won on cross-validated macro-F1 over Naive Bayes.
2. *Where did the data come from?* Hand-written seeds (drafted with AI help, reviewed by me) plus light variations; documented in the README and `data/LABELS.md`.
3. *Is 87% good?* It is a reasonable result on a small, hard-case-heavy dataset, but not proof of real-world performance; the challenge set (81%) and error analysis show where it breaks.
4. *What would you do with real data?* Collect real messages, label with two annotators, measure agreement, then try sentence embeddings or a fine-tuned transformer.
5. *Biggest weakness?* Style shortcuts (length, function words) and off-topic questions mislabelled as Inquiry.

## Exercises (do at least two)

1. In `train.py`, change `ngram_range=(1, 2)` to `(1, 1)` and re-run. What happens to the metrics, and why?
2. Remove the `qmark` replacement in `preprocess.py`. Which class suffers?
3. Add 10 new Feedback sentences that contain the word "not" to `seed_data.py`, rebuild, retrain. Does Feedback recall improve?
4. Replace `LogisticRegression` with `LinearSVC` (no probabilities, so adapt `predict.py`) and compare.
5. Find three sentences the model gets wrong, say why, and add similar training examples.
