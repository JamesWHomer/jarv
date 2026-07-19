"""Tolerant JSON extraction from model output.

Local and chat-completions models frequently wrap tool arguments or verdicts
in markdown fences or surround them with prose despite being prompted for
bare JSON. These helpers recover the intended object without guessing when
the text is genuinely ambiguous.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

_FENCED_BLOCK_RE = re.compile(r"```[A-Za-z0-9_-]*\s*\n(.*?)```", re.DOTALL)


def iter_json_objects(text: str) -> Iterator[object]:
    """Yield every JSON value decodable starting at a ``{`` in ``text``.

    A ``raw_decode`` scan at each brace survives markdown fences, prose
    before or after the object, and stray braces in surrounding prose. Every
    brace is tried — including braces nested inside an already-yielded object
    — so callers scanning for a specific shape (e.g. the auditor's verdict)
    still find it when a model wraps it in an outer envelope.
    """
    decoder = json.JSONDecoder()
    for match in re.finditer(r"{", text):
        try:
            value, _end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        yield value


def salvage_json_object(text: str) -> dict | None:
    """Best-effort recovery of a single JSON object from model output.

    Strips markdown code fences (preferring fenced content when present),
    then decodes at the first brace that yields a value. Returns ``None``
    when nothing decodes, the first decoded value is not an object, or a
    second top-level object follows the first — picking between two candidate
    argument objects would be worse than failing loudly.
    """
    if not text:
        return None
    fenced = _FENCED_BLOCK_RE.search(text)
    if fenced:
        text = fenced.group(1)
    try:
        whole = json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    else:
        # The text as a whole is valid JSON: a non-object (array, string…) is
        # the model's actual answer — cherry-picking a dict out of it would
        # be guessing.
        return whole if isinstance(whole, dict) else None
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if not isinstance(value, dict):
            return None
        remainder = text[start + end:]
        position = remainder.find("{")
        while position != -1:
            try:
                decoder.raw_decode(remainder[position:])
            except json.JSONDecodeError:
                position = remainder.find("{", position + 1)
                continue
            return None
        return value
    return None
