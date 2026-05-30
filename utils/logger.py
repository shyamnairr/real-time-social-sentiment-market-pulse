"""Consistent, colorized-ish console logging for all modules."""
import logging
import sys

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        _CONFIGURED = True
    return logging.getLogger(name)
