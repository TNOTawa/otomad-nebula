"""时间轴均匀采样规则（纯函数，便于测试与复现）。

规则（P0 参数：8 秒片段、最多 5 段）：

1. 视频不短于 8 秒时，片段长度固定 8 秒；不足 8 秒时整部视频作为唯一片段。
2. 能放下的互不重叠片段数为 floor(时长 / 片段长度)，并截断到最多 5 段。
3. 片段在时间轴上均匀分布：第 i 段的中心位于 时长*(i+0.5)/n。
4. 片段必须完全落在视频内；起始时间四舍五入到毫秒。

这些规则刻意保持简单、确定、无随机数：同一视频在任何机器上得到相同的切片计划。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
