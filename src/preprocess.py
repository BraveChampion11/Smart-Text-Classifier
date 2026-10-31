"""Text cleaning used for BOTH training and prediction.

Design choices (explained in the report):
  * Lower-case everything so "Refund" and "refund" are one feature.
  * Replace URLs and numbers with placeholder tokens (their exact value is
    noise, but "there is a link" / "there is a number" can be a signal).
  * Turn "?" and "!" into the words ` qmark ` / ` excl `. The default TF-IDF
    tokenizer throws punctuation away, but "?" is a useful hint for Inquiry.
  * Do NOT remove stopwords. Words such as "not", "how", "why", "can" decide
    the label ("not working" vs "working", "how do I" vs "I do").
"""
import re

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
NUM_RE = re.compile(r"\d+")
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = NUM_RE.sub(" numtoken ", text)
    text = text.replace("?", " qmark ").replace("!", " excl ")
    text = re.sub(r"[^a-z\s']", " ", text)  # keep letters and apostrophes
    return SPACE_RE.sub(" ", text).strip()


if __name__ == "__main__":
    for s in ["Where is my refund? It has been 2 weeks!", "Visit https://x.com NOW"]:
        print(repr(s), "->", repr(clean_text(s)))
