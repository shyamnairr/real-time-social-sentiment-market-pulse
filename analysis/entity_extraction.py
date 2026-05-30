"""
Entity extraction: find which tracked tickers a post is about.

Two signals:
  1. Cashtags: explicit "$AAPL" style mentions (highest precision).
  2. Bare symbols / company names: "AAPL" or "Apple" matched against our universe,
     with an ambiguous-word guard so "A", "ON", "GO" etc. don't false-trigger.

Returns the list of distinct tickers mentioned. The pipeline/collectors use this to
attach a ticker to each post (real posts may mention several -> one row per ticker).
"""
from __future__ import annotations

import re

from utils.tickers import (AMBIGUOUS_TICKERS, NAME_TO_TICKER, TICKER_TO_NAME,
                           is_valid_ticker)

_CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,6})\b")
_WORD_RE = re.compile(r"\b[A-Z]{2,6}\b")           # bare uppercase tokens
_NAME_RE = None  # built lazily


def _build_name_regex():
    global _NAME_RE
    if _NAME_RE is None:
        # match distinctive company names as whole words, case-insensitive
        names = sorted(NAME_TO_TICKER.keys(), key=len, reverse=True)
        pat = "|".join(re.escape(n) for n in names)
        _NAME_RE = re.compile(r"\b(" + pat + r")\b", re.IGNORECASE)
    return _NAME_RE


def extract_tickers(text: str) -> list[str]:
    if not text:
        return []
    found = set()

    # 1. cashtags -> high precision
    for m in _CASHTAG_RE.finditer(text):
        sym = m.group(1).upper()
        if is_valid_ticker(sym):
            found.add(sym)

    # 2. bare uppercase symbols (skip ambiguous)
    for m in _WORD_RE.finditer(text):
        sym = m.group(0).upper()
        if sym in AMBIGUOUS_TICKERS:
            continue
        if is_valid_ticker(sym):
            found.add(sym)

    # 3. company names
    for m in _build_name_regex().finditer(text):
        name = m.group(1).lower()
        tk = NAME_TO_TICKER.get(name)
        if tk:
            found.add(tk)

    return sorted(found)


def primary_ticker(text: str) -> str | None:
    """Return the single most likely ticker (first cashtag, else first match)."""
    tickers = extract_tickers(text)
    return tickers[0] if tickers else None
