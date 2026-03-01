"""
Console output helpers.

This module keeps CLI output robust across terminals with limited encodings
(for example, cp1252 on Windows).
"""

from __future__ import annotations

import sys
from typing import Any, TextIO

_ASCII_FALLBACKS = {
    "Δ": "d",
    "δ": "d",
    "½": "0.5",
    "²": "^2",
    "∞": "inf",
    "×": "x",
    "≡": "=",
    "→": "->",
    "←": "<-",
    "—": "-",
    "–": "-",
    "•": "*",
    "✓": "[ok]",
    "✗": "[x]",
    "★": "*",
    "☉": "sun",
    "♃": "jup",
    "°": " deg",
    "±": "+/-",
    "╔": "+",
    "╗": "+",
    "╚": "+",
    "╝": "+",
    "║": "|",
    "═": "=",
}


def _coerce_for_stream(text: str, stream: TextIO) -> str:
    """Return text encoded safely for the target stream."""
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return text
    try:
        text.encode(encoding)
        return text
    except UnicodeEncodeError:
        fallback = "".join(_ASCII_FALLBACKS.get(ch, ch) for ch in text)
        return fallback.encode(encoding, errors="replace").decode(encoding, errors="replace")


def safe_print(
    *args: Any,
    sep: str = " ",
    end: str = "\n",
    file: TextIO | None = None,
    flush: bool = False,
) -> None:
    """Print text without raising UnicodeEncodeError on restricted consoles."""
    stream = sys.stdout if file is None else file
    text = sep.join(str(a) for a in args)
    stream.write(_coerce_for_stream(text, stream) + end)
    if flush:
        stream.flush()


def configure_console_streams(errors: str = "replace") -> None:
    """Configure stdio streams to avoid UnicodeEncodeError on print()."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors=errors)
            except Exception:
                # Keep original stream behaviour if reconfiguration is unsupported.
                pass
