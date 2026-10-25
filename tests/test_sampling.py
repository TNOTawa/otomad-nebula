from __future__ import annotations

import math

import numpy as np
import pytest

from nebula.errors import SamplingError
from nebula.sampling import ClipPlan, CoverageClip, plan_clips, plan_full_coverage


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
        assert all(p.start_s >= 0.0 for p in plans)
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


# ──────────────────────────────────────────────────────────────────────────────
# v0.4 全片覆盖采样 plan_full_coverage（独立于 plan_clips）
# ──────────────────────────────────────────────────────────────────────────────


def test_coverage_short_video_single_window():
    plans = plan_full_coverage(3.0)
    assert len(plans) == 1
    plan = plans[0]
    assert isinstance(plan, CoverageClip)
    assert plan.index == 0
    assert plan.start_s == 0.0
    assert plan.duration_s == 3.0
    assert plan.valid_duration_s == 3.0
    assert plan.weight == 1.0
    assert plan.end_s == pytest.approx(3.0)


def test_coverage_exact_clip_length_single_window():
    plans = plan_full_coverage(8.0)
    assert len(plans) == 1
    assert plans[0].start_s == 0.0
    assert plans[0].duration_s == 8.0
    assert plans[0].weight == 1.0


def test_coverage_12s_two_windows():
    plans = plan_full_coverage(12.0)
    assert [p.start_s for p in plans] == [0.0, 4.0]
    assert all(p.duration_s == 8.0 for p in plans)
    assert plans[-1].end_s == pytest.approx(12.0, abs=1e-6)
    assert sum(p.weight for p in plans) == pytest.approx(1.0, abs=1e-6)


def test_coverage_40s_five_windows():
    plans = plan_full_coverage(40.0)
    assert len(plans) == 5
    assert [p.start_s for p in plans] == [0.0, 8.0, 16.0, 24.0, 32.0]
    assert plans[-1].end_s == pytest.approx(40.0, abs=1e-6)
    # 40s 恰好 5 个互不重叠的 8s 窗口：权重均等
    assert [p.weight for p in plans] == pytest.approx([0.2] * 5, abs=1e-6)


def test_coverage_180s_23_windows():
    plans = plan_full_coverage(180.0)
    assert len(plans) == 23
    assert plans[0].start_s == 0.0
    assert plans[-1].start_s == 172.0
    assert plans[-1].end_s == pytest.approx(180.0, abs=1e-6)
    assert all(p.duration_s == 8.0 and p.valid_duration_s == 8.0 for p in plans)
    assert sum(p.weight for p in plans) == pytest.approx(1.0, abs=1e-6)
    assert all(p.weight > 0.0 for p in plans)


def test_coverage_3600s_450_windows():
    plans = plan_full_coverage(3600.0)
    assert len(plans) == 450
    assert plans[-1].end_s == pytest.approx(3600.0, abs=1e-6)
    assert sum(p.weight for p in plans) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "duration",
    [1.0, 2.5, 7.9, 8.0, 8.5, 9.0, 12.0, 15.99, 16.1, 30.0, 40.0, 60.5, 100.0, 180.0, 300.0, 600.0],
)
def test_coverage_invariants(duration: float):
    plans = plan_full_coverage(duration)
    assert plans
    assert [p.index for p in plans] == list(range(len(plans)))

    expected_length = 8.0 if duration > 8.0 + 1e-9 else round(duration, 3)
    starts = np.asarray([p.start_s for p in plans], dtype=np.float64)
    ends = np.asarray([p.end_s for p in plans], dtype=np.float64)

    assert starts[0] == pytest.approx(0.0, abs=1e-9)
    assert ends[-1] == pytest.approx(duration, abs=1e-3)
    assert np.all(starts >= -1e-9)
    assert np.all(ends <= duration + 1e-6)
    assert np.all(np.diff(starts) >= -1e-9), "窗口起点必须单调非降"
    gaps = starts[1:] - ends[:-1]
    assert np.all(gaps <= 1e-6), f"窗口之间不能有未覆盖间隙: {gaps}"
    for plan in plans:
        assert plan.duration_s == pytest.approx(expected_length, abs=1e-6)
        assert plan.valid_duration_s == pytest.approx(plan.duration_s, abs=1e-6)
        assert plan.weight > 0.0
    assert sum(p.weight for p in plans) == pytest.approx(1.0, abs=1e-6)

    # 覆盖完整性：任意采样点至少被一个窗口覆盖
    for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
        point = duration * ratio
        assert any(p.start_s - 1e-9 <= point <= p.end_s + 1e-9 for p in plans), point


def test_coverage_weights_match_inverse_coverage_integral():
    duration = 180.0
    plans = plan_full_coverage(duration)
    windows = [(p.start_s, p.end_s) for p in plans]
    boundaries = sorted(
        {round(start, 9) for start, _end in windows}
        | {round(end, 9) for _start, end in windows}
    )
    integrals = [0.0] * len(windows)
    for left, right in zip(boundaries, boundaries[1:], strict=False):
        if right - left <= 1e-12:
            continue
        midpoint = (left + right) / 2.0
        covering = [
            index
            for index, (start, end) in enumerate(windows)
            if start - 1e-9 <= midpoint <= end + 1e-9
        ]
        assert covering
        for index in covering:
            integrals[index] += (right - left) / len(covering)
    assert sum(integrals) == pytest.approx(duration, abs=1e-6)
    for plan, integral in zip(plans, integrals, strict=True):
        assert plan.weight == pytest.approx(round(integral / duration, 9), abs=1e-9)


def test_coverage_each_second_inverse_coverage_mass_is_one():
    """未归一化的 1/覆盖计数 密度，任意一秒内的总权重为 1。"""

    duration = 180.0
    plans = plan_full_coverage(duration)
    for second_start in np.arange(0.0, duration, 1.0):
        midpoint = float(second_start) + 0.5
        covering = [
            plan
            for plan in plans
            if plan.start_s - 1e-9 <= midpoint <= plan.end_s + 1e-9
        ]
        assert covering, midpoint
        mass = sum(1.0 / len(covering) for _plan in covering)
        assert mass == pytest.approx(1.0)


@pytest.mark.parametrize("overlap", [0.5, 1.0, 2.0])
def test_coverage_overlap_lower_bound(overlap: float):
    plans = plan_full_coverage(180.0, overlap_seconds=overlap)
    expected = int(math.ceil((180.0 - 8.0) / (8.0 - overlap))) + 1
    assert len(plans) == expected
    for previous, following in zip(plans, plans[1:], strict=False):
        actual_overlap = previous.end_s - following.start_s
        assert actual_overlap >= overlap - 1e-6
        assert actual_overlap <= 8.0  # 重叠不可能超过窗口长度


def test_coverage_deterministic():
    assert plan_full_coverage(180.0) == plan_full_coverage(180.0)
    assert plan_full_coverage(37.21, overlap_seconds=1.5) == plan_full_coverage(
        37.21, overlap_seconds=1.5
    )


def test_coverage_does_not_change_plan_clips():
    assert plan_clips(12.0) == (ClipPlan(index=0, start_s=2.0, duration_s=8.0),)
    assert len(plan_clips(40.0)) == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration_s": 0},
        {"duration_s": -3},
        {"duration_s": float("nan")},
        {"duration_s": float("inf")},
        {"duration_s": 10.0, "clip_seconds": 0},
        {"duration_s": 10.0, "clip_seconds": float("nan")},
        {"duration_s": 10.0, "overlap_seconds": -1.0},
        {"duration_s": 10.0, "overlap_seconds": 8.0},
        {"duration_s": 10.0, "overlap_seconds": 9.0},
        {"duration_s": 10.0, "overlap_seconds": float("nan")},
        {"duration_s": 10.0, "min_duration_seconds": 0},
        {"duration_s": 0.5, "min_duration_seconds": 1.0},
    ],
)
def test_coverage_invalid_parameters_raise(kwargs):
    duration = kwargs.pop("duration_s")
    with pytest.raises(SamplingError):
        plan_full_coverage(duration, **kwargs)

