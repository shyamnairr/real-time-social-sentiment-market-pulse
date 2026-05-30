"""
Text cleaning for sentiment scoring.

Removes URLs, markdown, Reddit/HTML artifacts, emojis and excess whitespace while
preserving cashtags ($AAPL) and sentence content. DistilBERT base is uncased, so we
keep casing here and let the tokenizer lowercase.
"""
from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_MD_RE = re.compile(r"[*_`>#~]+")               # markdown emphasis / quotes / headers
_USER_SUB_RE = re.compile(r"(?:/?u/|/?r/)\w+")  # reddit u/ and r/ mentions
_WS_RE = re.compile(r"\s+")
# strip most emoji / pictographic ranges
_EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF" "\U00002B00-\U00002BFF" "\U0000FE00-\U0000FE0F" "]+",
    flags=re.UNICODE,
)


def clean_text(text: str) -> str:
    if not text:
        return ""
    t = str(text)
    t = _URL_RE.sub(" ", t)
    t = _HTML_RE.sub(" ", t)
    t = _USER_SUB_RE.sub(" ", t)
    t = _EMOJI_RE.sub(" ", t)
    t = _MD_RE.sub(" ", t)
    t = t.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
    t = _WS_RE.sub(" ", t).strip()
    return t


def clean_series(series):
    """Vectorized-ish cleaning over a pandas Series."""
    return series.fillna("").map(clean_text)
