"""Perceptual-hash image dedup (optional dependency, fail-open).

Design reference: mohobot ``mohobot/image_cache.py`` — reuse an already computed
result when a perceptually identical image arrives under different bytes (new
URL, re-encode, resize). Only the decision idea and the Hamming-distance
threshold are borrowed; the download / persistence / LRU code upstream is NOT
copied. See NOTICE.md for the upstream MIT notice.

Relation to the exact sha256 cache (0.7.1 behaviour):

* an exact key hit wins and costs nothing extra;
* only on an exact miss do we look for a similar image, and only inside the
  SAME ``(model versions, thresholds, decode version)`` scope — reusing a
  verdict produced under an older model or a looser gate would silently
  annotate with a stale decision;
* if ``imagehash`` / ``Pillow`` is unavailable, :func:`compute_phash` returns
  ``None`` and the whole path falls back to the exact cache unchanged.
"""

from __future__ import annotations

import io
from typing import Iterable, Optional, Tuple

try:  # Pillow ships with the worker overlay; imagehash is an optional add-on.
    from PIL import Image
except ImportError:  # pragma: no cover - exercised by the fallback tests
    Image = None

try:
    from imagehash import phash as _compute_phash
except ImportError:  # pragma: no cover - exercised by the fallback tests
    _compute_phash = None

PHASH_AVAILABLE = Image is not None and _compute_phash is not None

#: Strictly less than this Hamming distance counts as "the same picture".
#: 5 is the threshold used upstream; 0 would degenerate to exact matching.
PHASH_DISTANCE_THRESHOLD = 5


def compute_phash(raw: bytes) -> Optional[str]:
    """Return the pHash hex digest of ``raw``, or ``None`` when unavailable.

    Never raises: a corrupt / truncated image on the chat path must not break
    message processing. The caller treats ``None`` as "exact cache only".
    """
    if not PHASH_AVAILABLE or not raw:
        return None
    try:
        with Image.open(io.BytesIO(raw)) as img:
            return str(_compute_phash(img))
    except Exception:
        return None


def hamming_distance(a: str, b: str) -> int:
    """Hamming distance between two pHash hex strings (999 = incomparable)."""
    if not a or not b or len(a) != len(b):
        return 999
    try:
        return sum(bin(int(x, 16) ^ int(y, 16)).count("1") for x, y in zip(a, b))
    except ValueError:
        return 999


def similar_key(
    digest: Optional[str],
    entries: Iterable[Tuple[Tuple, str]],
    scope: Tuple,
) -> Optional[Tuple]:
    """Find a cached key whose pHash matches ``digest`` within the same scope.

    ``key[1:]`` is the cache scope: ``(versions, limits, DECODE_VERSION)``.
    A mismatch there is a hard reject, not a near miss.
    """
    if not digest:
        return None
    for key, cached in entries:
        if len(key) < 2 or key[1:] != scope:
            continue
        if hamming_distance(digest, cached) < PHASH_DISTANCE_THRESHOLD:
            return key
    return None
