"""时间轴采样规则（纯函数，便于测试与复现）。

包含两套独立规则：

1. plan_clips()：旧版“均匀抽最多 N 段”（v0.3 legacy），片段之间可以有空隙；
   规则见函数 docstring。旧行为与旧测试保持不变。
2. plan_full_coverage()：v0.4 全片覆盖采样，用几何均匀步长铺满整条时间轴：
   无未覆盖间隙、无越界、最后一个窗口精确锚定结尾；权重按 1/覆盖计数 积分
   归一，保证 ∑weight = 1。二者互不影响。

所有函数都刻意保持简单、确定、无随机数：同一视频在任何机器上得到相同的切片计划。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .errors import SamplingError


@dataclass(frozen=True)
class ClipPlan:
    """一个待截取的音画同步片段。"""

    index: int
    start_s: float
    duration_s: float

    @property
    def end_s(self) -> float:
        return round(self.start_s + self.duration_s, 6)


def plan_clips(
    duration_s: float,
    *,
    clip_seconds: float = 8.0,
    clips_per_work: int = 5,
    min_duration_seconds: float = 1.0,
) -> tuple[ClipPlan, ...]:
    """为一条作品时间轴生成均匀分布的片段计划。

    参数
    ----
    duration_s:
        视频总时长（秒）。
    clip_seconds:
        目标片段长度（秒）。
    clips_per_work:
        最多片段数。
    min_duration_seconds:
        允许处理的最短视频时长；更短则抛 SamplingError。
    """

    if not math.isfinite(duration_s) or duration_s <= 0:
        raise SamplingError(f"视频时长非法: {duration_s!r}")
    if not math.isfinite(clip_seconds) or clip_seconds <= 0:
        raise SamplingError(f"片段长度非法: {clip_seconds!r}")
    if clips_per_work < 1:
        raise SamplingError(f"片段数量上限非法: {clips_per_work!r}")
    if not math.isfinite(min_duration_seconds) or min_duration_seconds <= 0:
        raise SamplingError(f"最短时长非法: {min_duration_seconds!r}")
    if duration_s + 1e-9 < min_duration_seconds:
        raise SamplingError(
            f"视频时长 {duration_s:.3f}s 短于可处理下限 {min_duration_seconds:.3f}s"
        )

    if duration_s <= clip_seconds + 1e-9:
        # 极短视频：整段作为唯一片段
        actual_len = round(min(duration_s, clip_seconds), 3)
        return (ClipPlan(index=0, start_s=0.0, duration_s=actual_len),)

    actual_len = round(clip_seconds, 3)
    fits = int(math.floor(duration_s / actual_len + 1e-9))
    n = min(clips_per_work, max(1, fits))

    plans: list[ClipPlan] = []
    max_start = duration_s - actual_len
    for i in range(n):
        center = duration_s * (i + 0.5) / n
        start = center - actual_len / 2.0
        start = min(max(start, 0.0), max_start)
        start = round(start, 3)
        end_margin = round(duration_s - start, 3)
        duration = min(actual_len, end_margin)
        plans.append(ClipPlan(index=i, start_s=start, duration_s=duration))
    return tuple(plans)


@dataclass(frozen=True)
class CoverageClip:
    """全片覆盖采样中的一个时间窗口。

    duration_s:
        窗口长度，默认等于 clip_seconds；整片短于窗口时等于整片时长。
    valid_duration_s:
        有效时长（正常等于 duration_s；预留给未来短尾/无效帧场景）。
    weight:
        该窗口在整部作品时间轴上的权重份额；所有窗口 weight 之和为 1。
    """

    index: int
    start_s: float
    duration_s: float
    valid_duration_s: float
    weight: float

    @property
    def end_s(self) -> float:
        return round(self.start_s + self.duration_s, 6)


def _coverage_weights(windows: Sequence[tuple[float, float]], duration_s: float) -> list[float]:
    """按 1/覆盖计数 积分计算每个窗口的归一化权重。

    时间轴按所有窗口的 start/end 切成若干段，每段内覆盖计数 c(t) 为覆盖它的
    窗口数；窗口 i 的原始积分 S_i = ∫_{窗口 i} 1/c(t) dt，最终
    weight_i = S_i / duration_s（四舍五入到 1e-9）。由于
    Σ_i S_i = ∫_时间轴 c(t)/c(t) dt = duration_s，因此 Σ weight_i = 1。
    """

    events = sorted(
        {round(start, 9) for start, _end in windows} | {round(end, 9) for _start, end in windows}
    )
    integrals = [0.0] * len(windows)
    for left, right in zip(events, events[1:], strict=False):
        if right - left <= 1e-12:
            continue
        midpoint = (left + right) / 2.0
        covering = [
            index
            for index, (start, end) in enumerate(windows)
            if start - 1e-9 <= midpoint <= end + 1e-9
        ]
        if not covering:
            raise SamplingError(
                f"全片覆盖计划出现未覆盖间隙: [{left:.6f}, {right:.6f}]"
            )
        share = (right - left) / len(covering)
        for index in covering:
            integrals[index] += share
    return [round(value / duration_s, 9) for value in integrals]


def plan_full_coverage(
    duration_s: float,
    *,
    clip_seconds: float = 8.0,
    overlap_seconds: float = 0.0,
    min_duration_seconds: float = 1.0,
) -> tuple[CoverageClip, ...]:
    """为整条时间轴生成无间隙、无越界、末尾锚定的全覆盖窗口计划。

    算法（确定性）：

    1. 校验参数（与 plan_clips 一致，另要求 0 <= overlap_seconds < clip_seconds）；
    2. duration_s <= clip_seconds 时退化为单窗口 [0, duration_s]，weight = 1；
    3. 否则 n = ceil((duration - clip) / (clip - overlap)) + 1，
       step = (duration - clip) / (n - 1)，第 i 个起点为 i*step（四舍五入到毫秒），
       最后一个窗口起点精确锚定 duration - clip；
    4. 每个窗口长度都是 clip_seconds，因此无未覆盖间隙、无越界；
    5. 权重 = 对 1/覆盖计数(t) 的积分并归一（∑weight = 1）。

    参数
    ----
    duration_s:
        视频总时长（秒）。
    clip_seconds:
        窗口长度（秒）。
    overlap_seconds:
        相邻窗口的最小重叠时长（秒），必须满足 0 <= overlap < clip。
    min_duration_seconds:
        允许处理的最短视频时长；更短则抛 SamplingError。

    返回
    ----
    tuple[CoverageClip, ...]:
        按时间顺序排列的窗口；默认 8s/0 重叠下 180s → 23 个、40s → 5 个。

    异常
    ----
    SamplingError:
        参数非法、视频过短、或计算中检测到未覆盖间隙时抛出。
    """

    if not math.isfinite(duration_s) or duration_s <= 0:
        raise SamplingError(f"视频时长非法: {duration_s!r}")
    if not math.isfinite(clip_seconds) or clip_seconds <= 0:
        raise SamplingError(f"片段长度非法: {clip_seconds!r}")
    if not math.isfinite(overlap_seconds) or overlap_seconds < 0:
        raise SamplingError(f"重叠时长必须是非负有限数: {overlap_seconds!r}")
    if overlap_seconds >= clip_seconds:
        raise SamplingError(
            f"重叠时长必须小于片段长度: overlap={overlap_seconds!r} clip={clip_seconds!r}"
        )
    if not math.isfinite(min_duration_seconds) or min_duration_seconds <= 0:
        raise SamplingError(f"最短时长非法: {min_duration_seconds!r}")
    if duration_s + 1e-9 < min_duration_seconds:
        raise SamplingError(
            f"视频时长 {duration_s:.3f}s 短于可处理下限 {min_duration_seconds:.3f}s"
        )

    if duration_s <= clip_seconds + 1e-9:
        actual_len = round(min(duration_s, clip_seconds), 3)
        return (CoverageClip(index=0, start_s=0.0, duration_s=actual_len,
                             valid_duration_s=actual_len, weight=1.0),)

    delta = duration_s - clip_seconds
    step_max = clip_seconds - overlap_seconds
    count = int(math.ceil(delta / step_max - 1e-12)) + 1
    step = delta / (count - 1)
    last_start = duration_s - clip_seconds

    starts: list[float] = [0.0]
    for i in range(1, count - 1):
        start = round(i * step, 3)
        # 防御性约束：保持时间顺序、不越过结尾锚点、且与上一窗口不产生间隙。
        start = min(start, last_start)
        start = max(start, starts[-1])
        start = min(start, starts[-1] + clip_seconds)
        starts.append(start)
    starts.append(last_start)

    windows = [(start, start + clip_seconds) for start in starts]
    weights = _coverage_weights(windows, duration_s)
    return tuple(
        CoverageClip(
            index=index,
            start_s=starts[index],
            duration_s=clip_seconds,
            valid_duration_s=clip_seconds,
            weight=weights[index],
        )
        for index in range(count)
    )
