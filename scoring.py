"""Pure, dependency-free relevance-scoring primitives for StreamRAG.

This module imports only the standard library (no celery, qdrant, or
feedparser) so it can be unit-tested and reused in isolation. It provides
:func:`recency_weight`, the time-decay building block behind the roadmap's
"time-decayed / recency-weighted scoring" item.
"""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

# Weight returned when an entry has no usable ``published`` timestamp. Without a
# date we cannot assert recency, so such entries receive no time-decay boost
# instead of being optimistically treated as brand new. It is deliberately kept
# distinct in meaning from any value the decay curve yields for a real date.
UNKNOWN_DATE_WEIGHT = 0.0


def _as_aware_utc(value: datetime) -> datetime:
    """Return ``value`` as a timezone-aware UTC datetime.

    Naive datetimes (e.g. an RFC 2822 date written with the unknown-zone
    ``-0000`` marker) are assumed to already be expressed in UTC.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def recency_weight(
    published: str,
    now: datetime | None = None,
    half_life_hours: float = 24.0,
) -> float:
    """Score an entry's freshness in ``[0, 1]`` via exponential half-life decay.

    The weight is ``0.5 ** (age_hours / half_life_hours)``: it is ``1.0`` for an
    entry published exactly at ``now`` and halves every ``half_life_hours``.
    Entries dated in the future are clamped to ``1.0`` (treated as maximally
    fresh) and the result never leaves ``[0, 1]``.

    Args:
        published: An RFC 2822 date string as found in RSS ``published`` fields
            (e.g. ``"Mon, 06 Sep 2021 12:00:00 GMT"``). Empty or unparseable
            values yield :data:`UNKNOWN_DATE_WEIGHT`.
        now: Reference "current" time. Defaults to the current time in UTC.
            Naive datetimes are interpreted as UTC.
        half_life_hours: Age, in hours, at which the weight decays to ``0.5``.
            Must be positive.

    Returns:
        A float in ``[0, 1]``.

    Raises:
        ValueError: If ``half_life_hours`` is not positive.
    """
    if half_life_hours <= 0:
        raise ValueError("half_life_hours must be positive")

    if not published:
        return UNKNOWN_DATE_WEIGHT
    try:
        published_dt = parsedate_to_datetime(published)
    except (TypeError, ValueError):
        return UNKNOWN_DATE_WEIGHT
    if published_dt is None:  # defensive: some stdlib versions return None
        return UNKNOWN_DATE_WEIGHT

    reference = datetime.now(timezone.utc) if now is None else _as_aware_utc(now)
    published_dt = _as_aware_utc(published_dt)

    age_hours = (reference - published_dt).total_seconds() / 3600.0
    if age_hours <= 0:
        return 1.0
    weight = 0.5 ** (age_hours / half_life_hours)
    # Guard against floating-point drift nudging the result outside the range.
    return max(0.0, min(1.0, weight))
