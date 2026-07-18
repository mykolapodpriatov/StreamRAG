"""Tests for the pure ``recency_weight`` scoring primitive in scoring.py.

Every case pins ``now`` to a fixed timezone-aware datetime so the exponential
half-life decay is fully deterministic and independent of the wall clock or
network.
"""

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest

import scoring

NOW = datetime(2021, 9, 7, 12, 0, 0, tzinfo=timezone.utc)


def _rfc2822(dt: datetime) -> str:
    """Render a datetime the way an RSS ``published`` field would."""
    return format_datetime(dt)


def test_zero_age_scores_one():
    assert scoring.recency_weight(_rfc2822(NOW), now=NOW) == pytest.approx(1.0)


def test_one_half_life_scores_one_half():
    published = _rfc2822(NOW - timedelta(hours=24))
    weight = scoring.recency_weight(published, now=NOW, half_life_hours=24.0)
    assert weight == pytest.approx(0.5)


def test_two_half_lives_scores_one_quarter():
    published = _rfc2822(NOW - timedelta(hours=48))
    weight = scoring.recency_weight(published, now=NOW, half_life_hours=24.0)
    assert weight == pytest.approx(0.25)


def test_custom_half_life_halves_at_its_own_interval():
    published = _rfc2822(NOW - timedelta(hours=6))
    weight = scoring.recency_weight(published, now=NOW, half_life_hours=6.0)
    assert weight == pytest.approx(0.5)


def test_future_date_is_clamped_to_one():
    published = _rfc2822(NOW + timedelta(hours=10))
    assert scoring.recency_weight(published, now=NOW) == 1.0


def test_naive_and_unknown_zone_dates_are_treated_as_utc():
    # An RFC 2822 "-0000" date parses to a naive datetime; it must decay the
    # same as an explicit-UTC date one half-life old.
    naive = "Mon, 06 Sep 2021 12:00:00 -0000"  # 24h before NOW
    weight = scoring.recency_weight(naive, now=NOW, half_life_hours=24.0)
    assert weight == pytest.approx(0.5)


def test_very_old_date_stays_within_unit_interval():
    published = _rfc2822(NOW - timedelta(days=365))
    weight = scoring.recency_weight(published, now=NOW)
    assert 0.0 <= weight <= 1.0


@pytest.mark.parametrize("published", ["", "not a date", "2021-09-07"])
def test_unparseable_or_empty_returns_neutral_default(published):
    assert scoring.recency_weight(published, now=NOW) == scoring.UNKNOWN_DATE_WEIGHT


def test_non_positive_half_life_is_rejected():
    with pytest.raises(ValueError):
        scoring.recency_weight(_rfc2822(NOW), now=NOW, half_life_hours=0.0)
