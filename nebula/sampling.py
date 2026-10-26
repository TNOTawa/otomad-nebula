"""时间轴采样规则（纯函数，便于测试与复现）。

包含两套独立规则：

1. plan_clips()：旧版“均匀抽最多 N 段”（v0.3 legacy），片段之间可以有空隙；
   规则见函数 docstring。旧行为与旧测试保持不变。
2. plan_full_coverage()：v0.4 全片覆盖采样，用几何均匀步长铺满整条时间轴：
   无未覆盖间隙、无越界、最后一个窗口精确锚定结尾；权重由公开的
   coverage_weights() 按 1/覆盖计数 积分归一，保证 ∑weight = 1。二者互不影响。

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


def coverage_weights(
    windows: Sequence[tuple[float, float]], duration_s: float
) -> tuple[float, ...]:
    """按 1/覆盖计数 积分计算每个窗口的归一化覆盖权重。

    时间轴按所有窗口的 start/end 切成若干段，每段内覆盖计数 c(t) 为覆盖它的
    窗口数；窗口 i 的原始积分 S_i = ∫_{窗口 i} 1/c(t) dt，最终
    weight_i = S_i / duration_s（四舍五入到 1e-9）。由于窗口必须完整覆盖
    [0, duration_s]，Σ_i S_i = ∫_时间轴 c(t)/c(t) dt = duration_s，
    因此 Σ weight_i = 1。

    plan_full_coverage 使用本函数生成 CoverageClip.weight；v0.4.3.1 修复把窗口
    权重写进 dense_visual manifest，避免作品向量退化为窗口等权。

    参数
    ----
    windows:
        按时间顺序排列的 (start_s, end_s) 窗口序列，必须无间隙覆盖整条时间轴；
        允许相邻窗口重叠（重叠区按 1/覆盖计数 均摊）。
    duration_s:
        视频总时长（秒），必须大于 0 且与窗口覆盖范围一致。

    返回
    ----
    tuple[float, ...]:
        与 windows 等长、和为 1 的权重（四舍五入到 1e-9）。

    异常
    ----
    SamplingError:
        时长非法、窗口为空/格式非法/越界、存在未覆盖间隙，或总覆盖长度与
        duration_s 不一致时抛出。
    """

    if not math.isfinite(duration_s) or duration_s <= 0:
        raise SamplingError(f"视频时长非法: {duration_s!r}")
    if isinstance(windows, (str, bytes)) or not isinstance(windows, Sequence):
        raise SamplingError("windows 必须是 (start, end) 序列")
    if not windows:
        raise SamplingError("窗口列表不能为空")

    pairs: list[tuple[float, float]] = []
    for index, item in enumerate(windows):
        try:
            start, end = item
        except (TypeError, ValueError) as exc:
            raise SamplingError(f"第 {index} 个窗口必须是 (start, end) 二元组") from exc
        if isinstance(start, bool) or isinstance(end, bool):
            raise SamplingError(f"第 {index} 个窗口的 start/end 必须是数字")
        try:
            start_value = float(start)
            end_value = float(end)
        except (TypeError, ValueError) as exc:
            raise SamplingError(f"第 {index} 个窗口的 start/end 不是数字") from exc
        if not math.isfinite(start_value) or not math.isfinite(end_value):
            raise SamplingError(f"第 {index} 个窗口的 start/end 必须有限")
        if start_value < -1e-9:
            raise SamplingError(f"第 {index} 个窗口起点为负: {start_value!r}")
        if end_value <= start_value:
            raise SamplingError(
                f"第 {index} 个窗口结束必须大于起点: ({start_value!r}, {end_value!r})"
            )
        if end_value > duration_s + 1e-6:
            raise SamplingError(
                f"第 {index} 个窗口超出视频时长: end={end_value!r} duration={duration_s!r}"
            )
        pairs.append((max(start_value, 0.0), min(end_value, duration_s)))

    # 把边界（含时间轴两端）切成互不重叠的小段，按中点归属统计覆盖计数。
    events = sorted(
        {0.0, round(duration_s, 9)}
        | {round(start, 9) for start, _end in pairs}
        | {round(end, 9) for _start, end in pairs}
    )
    integrals = [0.0] * len(pairs)
    for left, right in zip(events, events[1:], strict=False):
        if right - left <= 1e-12:
            continue
        midpoint = (left + right) / 2.0
        covering = [
            index
            for index, (start, end) in enumerate(pairs)
            if start - 1e-9 <= midpoint <= end + 1e-9
        ]
        if not covering:
            raise SamplingError(
                f"窗口未覆盖整条时间轴，存在未覆盖间隙: [{left:.6f}, {right:.6f}]"
            )
        share = (right - left) / len(covering)
        for index in covering:
            integrals[index] += share
    total = sum(integrals)
    if abs(total - duration_s) > max(1e-6, duration_s * 1e-9):
        raise SamplingError(
            f"窗口总覆盖长度 {total:.9f}s 与视频时长 {duration_s:.9f}s 不一致"
        )
    return tuple(round(value / duration_s, 9) for value in integrals)


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
    weights = coverage_weights(windows, duration_s)
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
