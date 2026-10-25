from __future__ import annotations

import pytest

from nebula.errors import SamplingError
from nebula.sampling import ClipPlan, plan_clips


def test_long_video_five_non_overlapping_clips():
    plans = plan_clips(40.0)
    assert len(plans) == 5
    assert [p.start_s for p in plans] == [0.0, 8.0, 16.0, 24.0, 32.0]
    assert all(p.duration_s == 8.0 for p in plans)
    assert [p.index for p in plans] == [0, 1, 2, 3, 4]


def test_short_video_gets_centered_single_clip():
    plans = plan_clips(12.0)
    assert plans == (ClipPlan(index=0, start_s=2.0, duration_s=8.0),)


def test_exactly_one_clip_length():
    plans = plan_clips(8.0)
    assert plans == (ClipPlan(index=0, start_s=0.0, duration_s=8.0),)


def test_ultra_short_video_uses_whole_video():
    plans = plan_clips(7.5)
    assert plans == (ClipPlan(index=0, start_s=0.0, duration_s=7.5),)
    plans = plan_clips(3.0)
    assert plans == (ClipPlan(index=0, start_s=0.0, duration_s=3.0),)


def test_two_clips():
    assert [p.start_s for p in plan_clips(16.0)] == [0.0, 8.0]
    assert [p.start_s for p in plan_clips(17.0)] == [0.25, 8.75]


def test_clips_stay_inside_video_and_are_sorted():
    for duration in [1.0, 2.5, 8.5, 9.0, 15.99, 16.1, 30.0, 45.7, 90.0, 300.0]:
        plans = plan_clips(duration)
        assert plans, duration
        assert all(0.0 <= p.start_s for p in plans)
        assert all(p.start_s + p.duration_s <= duration + 1e-6 for p in plans)
        assert all(p.duration_s > 0 for p in plans)
        assert [p.start_s for p in plans] == sorted(p.start_s for p in plans)
        assert [p.index for p in plans] == list(range(len(plans)))


def test_clip_count_never_exceeds_limit():
    for duration in [100.0, 500.0, 3600.0]:
        assert len(plan_clips(duration)) == 5


def test_deterministic():
    assert plan_clips(73.21) == plan_clips(73.21)


def test_custom_parameters():
    plans = plan_clips(30.0, clip_seconds=5.0, clips_per_work=3)
    assert len(plans) == 3
    assert all(p.duration_s == 5.0 for p in plans)


def test_below_minimum_duration_raises():
    with pytest.raises(SamplingError):
        plan_clips(0.5, min_duration_seconds=1.0)


@pytest.mark.parametrize("duration", [0, -3, float("nan"), float("inf")])
def test_invalid_duration_raises(duration):
    with pytest.raises(SamplingError):
        plan_clips(duration)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"clip_seconds": 0},
        {"clips_per_work": 0},
        {"min_duration_seconds": 0},
    ],
)
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(SamplingError):
        plan_clips(10.0, **kwargs)
