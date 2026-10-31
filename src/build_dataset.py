"""Build data/dataset.csv from the hand-written seeds in seed_data.py.

Each seed sentence becomes one group. We add a few light variations of it
(greeting / sign-off wrapper, lower-casing, dropped final punctuation, one
adjacent-letter typo). The wrappers are the SAME for every class, so they
carry no label information.

Why a `group` column?
    Variations of the same seed are near-duplicates. If one variant landed in
    the training set and another in the test set, the test score would be
    inflated (data leakage). train.py splits by `group`, so all variants of a
    seed stay on the same side of the split.

Also writes data/challenge_set.csv: 32 NEW hand-written sentences (different
wording from the seeds) that are never used for training or model selection.
They give a more honest estimate of how the model behaves on unseen text.

Run:  python src/build_dataset.py
"""
import random
from pathlib import Path

import pandas as pd

from seed_data import SEEDS

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RANDOM_SEED = 42
VARIANTS_PER_SEED = 2  # extra variants besides the original sentence

PREFIXES = ["Hi, ", "Hello team, ", "Hey, ", "Excuse me, ", "Dear support, ", "To whom it may concern: "]
SUFFIXES = [" Thanks.", " Thank you.", " Please help.", " Regards.", " Waiting for your reply."]

CHALLENGE = {
    "Complaint": [
        "The courier left my package outside in the rain and now it is ruined.",
        "I have been on hold for 45 minutes and still no one has answered.",
        "Why was my subscription renewed when I clearly cancelled it last month?",
        "The technician never showed up even though I took a day off work.",
        "I would be thankful if someone could look into why my account was suspended.",
        "Third time my payment failed but the money left my account.",
        "your app deleted all my saved addresses after the update, very annoying",
        "The room was dirty and the manager did not care.",
    ],
    "Inquiry": [
        "Could you tell me when the next batch of admissions opens?",
        "How much does express shipping cost for two kilograms?",
        "need invoice copy for march",
        "Is it possible to switch to the yearly plan?",
        "Do you have any branch open on Sunday",
        "Which documents should I bring for the verification?",
        "What happens to my data if I delete my account?",
        "Where can I track my refund?",
    ],
    "Feedback": [
        "The new update looks great, though the font is a little small.",
        "Support agent was kind and explained everything patiently.",
        "I wish there were an option to export reports as PDF.",
        "Prices are a bit steep but I understand the quality is good.",
        "The tutorial videos are clear, more of them would be welcome.",
        "Not a fan of the new colour scheme, but everything works.",
        "great job on the faster loading times",
        "Maybe add a search bar on the home page?",
    ],
    "Other": [
        "Good afternoon everyone",
        "Thanks a lot, bye!",
        "You have been selected for a free vacation, call now",
        "What is the tallest mountain in the world?",
        "Haha that was funny",
        "Testing testing one two three",
        "Anyone know a good place for biryani in town?",
        "Wishing everyone a blessed Friday",
    ],
}


def add_typo(text: str, rng: random.Random) -> str:
    """Swap two adjacent letters inside one longer word (simulates a typo)."""
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not candidates:
        return text
    i = rng.choice(candidates)
    w = words[i]
    j = rng.randrange(1, len(w) - 2)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def make_variant(text: str, rng: random.Random) -> str:
    out = text
    if rng.random() < 0.5:
        out = rng.choice(PREFIXES) + out[0].lower() + out[1:]
    if rng.random() < 0.4:
        out = out + rng.choice(SUFFIXES)
    if rng.random() < 0.3:
        out = out.lower()
    if rng.random() < 0.25:
        out = out.rstrip(".!")  # drop final full stop / exclamation (keep '?')
    if rng.random() < 0.2:
        out = add_typo(out, rng)
    return out


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    rows, group_id = [], 0
    for label, seeds in SEEDS.items():
        for seed in seeds:
            group_id += 1
            variants = [seed]  # a list (not a set) keeps the order reproducible
            tries = 0
            while len(variants) < 1 + VARIANTS_PER_SEED and tries < 20:
                v = make_variant(seed, rng)
                if v not in variants:
                    variants.append(v)
                tries += 1
            for v in variants:
                rows.append({"text": v.strip(), "label": label, "group": group_id})

    df = pd.DataFrame(rows).drop_duplicates(subset="text")
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_DIR / "dataset.csv", index=False, encoding="utf-8")

    ch = pd.DataFrame(
        [{"text": t, "label": lab} for lab, texts in CHALLENGE.items() for t in texts]
    )
    overlap = set(ch["text"].str.lower()) & set(df["text"].str.lower())
    assert not overlap, f"Challenge set overlaps training data: {overlap}"
    ch.to_csv(DATA_DIR / "challenge_set.csv", index=False, encoding="utf-8")

    print(f"dataset.csv: {len(df)} rows, {df['group'].nunique()} seed groups")
    print(df["label"].value_counts().to_string())
    print(f"challenge_set.csv: {len(ch)} rows")


if __name__ == "__main__":
    main()
