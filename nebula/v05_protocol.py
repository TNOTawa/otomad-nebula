"""v0.5-beta 协议 v2：固定 K 候选、双向偏移、周期/近重复独立集与旧 checkpoint 重评。

背景：旧 beta 协议候选数为 1 正 + 最多 5 负（可变）、负样本只取未来偏移、近重复候选直接
删除，导致 random Top1 不等于 25%、Recall@5 无信息量、周期画面无法单独评估。本模块在
**不修改 nebula/v05_beta.py 旧行为** 的前提下定义并实现新协议：

1. 固定 K=6 主协议：1 正 + 5 负；负样本从同作品双向时间偏移
   ±1,±2,±4,±8,±16 bins（0.5s/bin）可用集合中确定性采样；无足够候选的 anchor 跳过并记录
   candidate_count 分布；主指标 Top-1 / Recall@1（K=6 的 Recall@5 明确标注无信息量）。
2. 周期/近重复独立集：anchor 与某偏移候选的视觉或音频余弦 > 0.95 时，整个 anchor 进入
   periodic 集（不删除、不进主训练/主测试），在 periodic 集单独评估（候选数可变，报告
   chance=mean(1/K)）。
3. K=32 retrieval：1 正 + 31 负，负样本从同作品其余 bin 中确定性采样（排除近重复），
   报告 Top-1 / Recall@1/5/10 与 chance（1/32、5/32、10/32）。
4. 划分固定为 beta 的 canonical work_id 74/21/12（从 logs/v05_beta_summary.json 读取并校验），
   train/val/test 不重叠；anchor 级记录（work_id、anchor bin/time、candidate bins、label、
   offset 方向、periodic 标志、split）保存到 data/v05_protocol/v2。
5. 在同一协议下重评旧 MLP checkpoints（v05_beta_42/43/44）以及 random、frozen projection、
   simple_sync；输出 logs/v05_protocol_v2_metrics.json + CSV，不写 DB、不建 map、不训练新模型。
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import v05_beta, v05_data
from .config import Config
from .errors import NebulaError
from .util import now_utc_iso

LOGGER = logging.getLogger(__name__)

PROTOCOL_VERSION = "v2"
PROTOCOL_DIRNAME = "v05_protocol"
TOKENS_VERSION = "v05_beta_v1"
MAIN_OFFSETS: tuple[int, ...] = (-16, -8, -4, -2, -1, 1, 2, 4, 8, 16)
MAIN_K = 6
RETRIEVAL_K = 32
MAIN_NEGATIVES = MAIN_K - 1
RETRIEVAL_NEGATIVES = RETRIEVAL_K - 1
NEAR_DUP_COSINE = 0.95
BIN_SECONDS = 0.5
SPLIT_SEED = 42
BETA_SPLIT_COUNTS = (74, 21, 12)
DEFAULT_BATCH_SIZE = 512
DEFAULT_BOOTSTRAP = 1000
PERIODIC_CANDIDATE_EVAL_LIMIT = 10000
SPLIT_CODES = {"train": 0, "val": 1, "test": 2}
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}
ANCHOR_FILENAME = "anchors.npz"
MANIFEST_FILENAME = "manifest.json"


class V05ProtocolError(NebulaError):
    """协议 v2 的可预期错误（split/checkpoint/数据不匹配等）。"""


@dataclass(frozen=True)
class ProtocolConfig:
    version: str = PROTOCOL_VERSION
    tokens_version: str = TOKENS_VERSION
    audio_version: str | None = None
    visual_encoder: str = "clip"
    seed: int = SPLIT_SEED
    near_dup: float = NEAR_DUP_COSINE
    main_k: int = MAIN_K
    retrieval_k: int = RETRIEVAL_K
    batch_size: int = DEFAULT_BATCH_SIZE
    device: str = "auto"
    bootstrap: int = DEFAULT_BOOTSTRAP
    evaluation_split: str = "test"
    checkpoint_paths: tuple[str, ...] = ()
    force: bool = False

    def validate(self) -> "ProtocolConfig":
        if str(self.version) != PROTOCOL_VERSION:
            raise V05ProtocolError(
                f"协议版本必须是 {PROTOCOL_VERSION!r}，收到 {self.version!r}"
            )
        if self.visual_encoder not in ("clip", "dino"):
            raise V05ProtocolError("visual_encoder 必须是 clip 或 dino")
        if int(self.main_k) < 2:
            raise V05ProtocolError("main_k 必须 >= 2")
        if int(self.retrieval_k) <= int(self.main_k):
            raise V05ProtocolError("retrieval_k 必须大于 main_k")
        if not 0.0 <= float(self.near_dup) <= 1.0:
            raise V05ProtocolError("near_dup 必须在 [0,1]")
        if int(self.batch_size) < 1 or int(self.bootstrap) < 0:
            raise V05ProtocolError("batch_size >= 1 且 bootstrap >= 0")
        if self.device not in ("auto", "cpu", "cuda"):
            raise V05ProtocolError("device 必须是 auto/cpu/cuda")
        if self.evaluation_split not in SPLIT_CODES:
            raise V05ProtocolError("evaluation_split 必须是 train/val/test")
        return self


@dataclass
class AnchorRecord:
    work_id: int
    anchor_bin: int
    anchor_time: float
    split: str
    group: str
    label: int
    candidate_bins: np.ndarray
    candidate_offsets: np.ndarray
    available_negatives: int


@dataclass
class PeriodicCandidateRecord:
    """主协议中被近重复规则排除的候选（进入独立 periodic 候选集，不删除）。"""

    work_id: int
    anchor_bin: int
    candidate_bin: int
    offset: int
    split: str


@dataclass
class RetrievalRecord:
    work_id: int
    anchor_bin: int
    anchor_time: float
    split: str
    label: int
    candidate_bins: np.ndarray
    candidate_offsets: np.ndarray
    available_negatives: int = 0


@dataclass
class ProtocolBundle:
    pairs: dict[int, v05_beta.WorkPair]
    split: dict[str, tuple[int, ...]]
    main_anchors: dict[str, list[AnchorRecord]]
    periodic_anchors: list[AnchorRecord]
    periodic_candidates: list[PeriodicCandidateRecord]
    retrieval_anchors: list[RetrievalRecord]
    manifest: dict[str, Any]
    data_info: dict[str, Any]


@dataclass
class LoadedCheckpoint:
    name: str
    path: Path
    model: Any
    info: dict[str, Any]


# ---------------------------------------------------------------- split / 数据加载


def load_beta_split(config: Config) -> dict[str, tuple[int, ...]]:
    """读取 beta 官方 split（74/21/12），保证与历史结果可比。"""

    summary_path = config.paths.logs / "v05_beta_summary.json"
    if not summary_path.is_file():
        raise V05ProtocolError(
            f"找不到 beta 官方 split: {summary_path}；协议 v2 需要固定 74/21/12 划分"
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise V05ProtocolError(f"读取 beta summary 失败（{summary_path}）: {exc}") from exc
    split = (summary.get("dataset") or {}).get("split") or {}
    train = tuple(int(value) for value in split.get("train_work_ids", ()))
    val = tuple(int(value) for value in split.get("val_work_ids", ()))
    test = tuple(int(value) for value in split.get("test_work_ids", ()))
    counts = (len(train), len(val), len(test))
    if counts != BETA_SPLIT_COUNTS:
        raise V05ProtocolError(
            f"beta 官方 split 数量 {counts} 与协议约定 {BETA_SPLIT_COUNTS} 不一致"
        )
    all_ids = train + val + test
    if len(set(all_ids)) != len(all_ids):
        raise V05ProtocolError("beta 官方 split 存在重叠 work_id")
    return {"train": tuple(sorted(train)), "val": tuple(sorted(val)), "test": tuple(sorted(test))}


def _resolve_audio_version(
    config: Config, requested: str | None, beta_summary_version: str | None
) -> tuple[str, Path, dict[str, Any] | None]:
    version = requested or beta_summary_version
    return v05_beta.resolve_audio_version(config, version)


def load_beta_audio_version(config: Config) -> str:
    summary_path = config.paths.logs / "v05_beta_summary.json"
    if not summary_path.is_file():
        raise V05ProtocolError(f"找不到 beta summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    version = ((summary.get("dataset") or {}).get("audio_version")) or (
        summary.get("config") or {}
    ).get("audio_version")
    if not version:
        raise V05ProtocolError("beta summary 缺少 audio_version")
    return str(version)


def build_pairs(
    config: Config,
    work_ids: Sequence[int],
    *,
    tokens_version: str,
    audio_version: str,
    visual_encoder: str,
    audio_manifest: Mapping[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[int, v05_beta.WorkPair], list[int]]:
    """为指定 work_id 构建配对序列；返回 (pairs, failed_ids)。"""

    pairs: dict[int, v05_beta.WorkPair] = {}
    failed: list[int] = []
    total = len(work_ids)
    for index, work_id in enumerate(work_ids, start=1):
        tokens = v05_data.load_work_tokens(config, tokens_version, int(work_id))
        if tokens is None:
            failed.append(int(work_id))
            LOGGER.warning("缺少 token: work_id=%s", work_id)
            continue
        audio = v05_beta.load_audio_bins(
            config,
            audio_version,
            int(work_id),
            version_manifest=audio_manifest,
        )
        if audio is None:
            failed.append(int(work_id))
            LOGGER.warning("缺少 audio: work_id=%s", work_id)
            continue
        pair, details = v05_beta.build_work_pair(
            tokens, audio, work_id=int(work_id), visual_encoder=visual_encoder
        )
        if pair is None:
            failed.append(int(work_id))
            LOGGER.warning("配对失败 work_id=%s: %s", work_id, details.get("reason"))
            continue
        pairs[int(work_id)] = pair
        if progress:
            progress(
                f"[{index}/{total}] work_id={work_id} 配对 bin={pair.n_bins} "
                f"覆盖率={pair.coverage.get('matched_ratio', 0):.3f}"
            )
    return pairs, failed


# ---------------------------------------------------------------- 候选构造（纯函数，可单测）


def _deterministic_rng(seed: int, work_id: int, anchor: int, kind: str) -> np.random.Generator:
    # 稳定组合（不依赖 Python hash 随机化）：kind 转成整数标签
    kind_map = {"main": 1, "periodic": 2, "retrieval": 3}
    return np.random.default_rng(
        [int(seed), int(work_id), int(anchor), kind_map.get(kind, 0)]
    )


def _cosine_pair(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(first, second) / denominator)


def _record_from_candidates(
    *,
    work_id: int,
    anchor: int,
    anchor_time: float,
    split: str,
    group: str,
    candidate_bins: Sequence[int],
    available_negatives: int,
    rng: np.random.Generator,
) -> AnchorRecord:
    positive_index = int(candidate_bins[0])
    bins_array = np.asarray([int(value) for value in candidate_bins], dtype=np.int64)
    permutation = rng.permutation(bins_array.size)
    ordered_bins = bins_array[permutation]
    label = int(np.flatnonzero(ordered_bins == positive_index)[0])
    offsets = ordered_bins - int(anchor)
    return AnchorRecord(
        work_id=int(work_id),
        anchor_bin=int(anchor),
        anchor_time=float(anchor_time),
        split=str(split),
        group=str(group),
        label=label,
        candidate_bins=ordered_bins.astype(np.int64),
        candidate_offsets=offsets.astype(np.int64),
        available_negatives=int(available_negatives),
    )


def build_anchor_records(
    pair: v05_beta.WorkPair,
    *,
    split: str,
    seed: int,
    near_dup: float,
    main_k: int = MAIN_K,
    retrieval_k: int = RETRIEVAL_K,
    build_retrieval: bool = False,
) -> tuple[
    list[AnchorRecord],
    list[AnchorRecord],
    list[PeriodicCandidateRecord],
    list[RetrievalRecord],
    dict[str, Any],
]:
    """为一部作品构造 v2 主集/periodic 集/候选（+可选 retrieval K=32）记录。"""

    if int(main_k) < 2:
        raise V05ProtocolError("main_k 必须 >= 2")
    n_bins = pair.n_bins
    main_records: list[AnchorRecord] = []
    periodic_records: list[AnchorRecord] = []
    periodic_candidates: list[PeriodicCandidateRecord] = []
    retrieval_records: list[RetrievalRecord] = []
    stats: dict[str, Any] = {
        "anchors": n_bins,
        "main_kept": 0,
        "periodic_anchors": 0,
        "periodic_candidates": 0,
        "main_skipped_insufficient": 0,
        "time_ambiguous": 0,
        "out_of_range": 0,
        "available_negatives_hist": {},
        "periodic_candidates_hist": {},
        "periodic_candidate_offset_hist": {},
        "retrieval_kept": 0,
        "retrieval_skipped_insufficient": 0,
        "retrieval_valid_candidates_hist": {},
    }

    def _bump(key: str, value: int) -> None:
        hist = stats[key]
        hist[str(int(value))] = int(hist.get(str(int(value)), 0)) + 1

    for anchor in range(n_bins):
        visual_anchor = pair.visual[anchor]
        audio_anchor = pair.audio[anchor]
        visual_time = float(pair.times[anchor])
        available: list[tuple[int, int]] = []
        periodic: list[tuple[int, int]] = []
        for offset in MAIN_OFFSETS:
            target = anchor + int(offset)
            if target < 0 or target >= n_bins:
                stats["out_of_range"] += 1
                continue
            expected = float(abs(offset)) * BIN_SECONDS
            gap = abs(float(pair.times[target] - pair.times[anchor]))
            if abs(gap - expected) > BIN_SECONDS * 0.5 + 1e-6:
                stats["time_ambiguous"] += 1
                continue
            near = (
                _cosine_pair(visual_anchor, pair.visual[target]) > float(near_dup)
                or _cosine_pair(audio_anchor, pair.audio[target]) > float(near_dup)
            )
            if near:
                periodic.append((int(offset), int(target)))
            else:
                available.append((int(offset), int(target)))
        _bump("available_negatives_hist", len(available))
        # 近重复候选单独进入 periodic candidate 集（不删除、也不进入主 K=6 候选池）
        for offset, target in periodic:
            periodic_candidates.append(
                PeriodicCandidateRecord(
                    work_id=int(pair.work_id),
                    anchor_bin=int(anchor),
                    candidate_bin=int(target),
                    offset=int(offset),
                    split=str(split),
                )
            )
            stats["periodic_candidates"] += 1
            offset_hist = stats["periodic_candidate_offset_hist"]
            offset_key = str(int(offset))
            offset_hist[offset_key] = int(offset_hist.get(offset_key, 0)) + 1
        if len(available) < int(main_k) - 1:
            # 不足 5 个非近重复负样本：若存在 periodic 候选，则整个 anchor 进 periodic 集；
            # 否则记为 insufficient 跳过（不伪造候选）。
            if periodic:
                _bump("periodic_candidates_hist", len(periodic))
                rng = _deterministic_rng(seed, pair.work_id, anchor, "periodic")
                candidate_bins = [anchor] + [target for _, target in periodic]
                periodic_records.append(
                    _record_from_candidates(
                        work_id=pair.work_id,
                        anchor=anchor,
                        anchor_time=visual_time,
                        split=split,
                        group="periodic",
                        candidate_bins=candidate_bins,
                        available_negatives=len(periodic),
                        rng=rng,
                    )
                )
                stats["periodic_anchors"] += 1
            else:
                stats["main_skipped_insufficient"] += 1
            continue
        rng = _deterministic_rng(seed, pair.work_id, anchor, "main")
        chosen_indices = rng.choice(len(available), size=int(main_k) - 1, replace=False)
        chosen = [available[int(index)] for index in chosen_indices]
        candidate_bins = [anchor] + [target for _offset, target in chosen]
        main_records.append(
            _record_from_candidates(
                work_id=pair.work_id,
                anchor=anchor,
                anchor_time=visual_time,
                split=split,
                group="main",
                candidate_bins=candidate_bins,
                available_negatives=len(available),
                rng=rng,
            )
        )
        stats["main_kept"] += 1

        if build_retrieval:
            valid_targets = [
                target
                for target in range(n_bins)
                if target != anchor
                and _cosine_pair(visual_anchor, pair.visual[target]) <= float(near_dup)
                and _cosine_pair(audio_anchor, pair.audio[target]) <= float(near_dup)
            ]
            _bump("retrieval_valid_candidates_hist", len(valid_targets))
            if len(valid_targets) < int(retrieval_k) - 1:
                stats["retrieval_skipped_insufficient"] += 1
                continue
            retrieval_rng = _deterministic_rng(seed, pair.work_id, anchor, "retrieval")
            chosen_targets = retrieval_rng.choice(
                len(valid_targets), size=int(retrieval_k) - 1, replace=False
            )
            candidate_bins = [anchor] + [valid_targets[int(index)] for index in chosen_targets]
            record = _record_from_candidates(
                work_id=pair.work_id,
                anchor=anchor,
                anchor_time=visual_time,
                split=split,
                group="retrieval",
                candidate_bins=candidate_bins,
                available_negatives=len(valid_targets),
                rng=retrieval_rng,
            )
            retrieval_records.append(
                RetrievalRecord(
                    work_id=record.work_id,
                    anchor_bin=record.anchor_bin,
                    anchor_time=record.anchor_time,
                    split=record.split,
                    label=record.label,
                    candidate_bins=record.candidate_bins,
                    candidate_offsets=record.candidate_offsets,
                    available_negatives=len(valid_targets),
                )
            )
            stats["retrieval_kept"] += 1
    return (
        main_records,
        periodic_records,
        periodic_candidates,
        retrieval_records,
        stats,
    )


def build_protocol_bundle(
    pairs: Mapping[int, v05_beta.WorkPair],
    split: Mapping[str, Sequence[int]],
    *,
    config: ProtocolConfig,
    data_info: Mapping[str, Any],
) -> ProtocolBundle:
    """从配对序列构造协议 v2 bundle（纯函数，测试可用合成 pairs）。"""

    cfg = config.validate()
    main_anchors: dict[str, list[AnchorRecord]] = {"train": [], "val": [], "test": []}
    periodic_anchors: list[AnchorRecord] = []
    periodic_candidates: list[PeriodicCandidateRecord] = []
    retrieval_anchors: list[RetrievalRecord] = []
    per_split_stats: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "val", "test"):
        split_stats = {
            "works": 0,
            "anchors": 0,
            "main_kept": 0,
            "periodic_anchors": 0,
            "periodic_candidates": 0,
            "main_skipped_insufficient": 0,
            "time_ambiguous": 0,
            "out_of_range": 0,
            "retrieval_kept": 0,
            "retrieval_skipped_insufficient": 0,
            "available_negatives_hist": {},
            "periodic_candidates_hist": {},
            "periodic_candidate_offset_hist": {},
            "retrieval_valid_candidates_hist": {},
        }
        for work_id in split[split_name]:
            pair = pairs.get(int(work_id))
            if pair is None:
                raise V05ProtocolError(f"split={split_name} 缺少 work_id={work_id} 的配对序列")
            build_retrieval = split_name == cfg.evaluation_split
            (
                main_records,
                periodic_records,
                periodic_candidate_records,
                retrieval_records,
                stats,
            ) = build_anchor_records(
                pair,
                split=split_name,
                seed=cfg.seed,
                near_dup=cfg.near_dup,
                main_k=cfg.main_k,
                retrieval_k=cfg.retrieval_k,
                build_retrieval=build_retrieval,
            )
            main_anchors[split_name].extend(main_records)
            periodic_anchors.extend(periodic_records)
            periodic_candidates.extend(periodic_candidate_records)
            retrieval_anchors.extend(retrieval_records)
            split_stats["works"] += 1
            for key in (
                "anchors",
                "main_kept",
                "periodic_anchors",
                "periodic_candidates",
                "main_skipped_insufficient",
                "time_ambiguous",
                "out_of_range",
                "retrieval_kept",
                "retrieval_skipped_insufficient",
            ):
                split_stats[key] += int(stats.get(key, 0))
            for hist_key in (
                "available_negatives_hist",
                "periodic_candidates_hist",
                "periodic_candidate_offset_hist",
                "retrieval_valid_candidates_hist",
            ):
                merged = split_stats[hist_key]
                for label, count in stats.get(hist_key, {}).items():
                    merged[str(label)] = int(merged.get(str(label), 0)) + int(count)
        per_split_stats[split_name] = split_stats
    total_main = sum(len(values) for values in main_anchors.values())
    total_available = sum(
        int(per_split_stats[name]["anchors"]) for name in per_split_stats
    )
    merged_available_hist: dict[str, int] = {}
    for name in per_split_stats:
        for label, count in per_split_stats[name]["available_negatives_hist"].items():
            merged_available_hist[str(label)] = int(
                merged_available_hist.get(str(label), 0)
            ) + int(count)
    total_periodic = sum(
        int(per_split_stats[name]["periodic_anchors"]) for name in per_split_stats
    )
    total_skipped_insufficient = sum(
        int(per_split_stats[name]["main_skipped_insufficient"]) for name in per_split_stats
    )
    total_time_ambiguous = sum(
        int(per_split_stats[name]["time_ambiguous"]) for name in per_split_stats
    )
    total_out_of_range = sum(
        int(per_split_stats[name]["out_of_range"]) for name in per_split_stats
    )
    eval_anchor_total = int(per_split_stats[cfg.evaluation_split]["anchors"])
    distribution = {
        "main": {
            "k": int(cfg.main_k),
            "anchors_total": total_available,
            "kept": total_main,
            "kept_ratio": float(total_main / total_available) if total_available else None,
            "skipped_insufficient_or_periodic": total_available - total_main,
            "candidate_count_hist": {str(cfg.main_k): total_main},
            "available_negatives_hist": dict(
                sorted(merged_available_hist.items(), key=lambda item: int(item[0]))
            ),
            "main_skipped_insufficient": total_skipped_insufficient,
            "time_ambiguous": total_time_ambiguous,
            "out_of_range": total_out_of_range,
            "periodic_anchors": total_periodic,
            "periodic_anchor_ratio": (
                float(total_periodic / total_available) if total_available else None
            ),
        },
        "periodic": {
            "anchors": len(periodic_anchors),
            "candidates": len(periodic_candidates),
            "candidate_ratio": (
                float(len(periodic_candidates) / total_available)
                if total_available
                else None
            ),
            "candidate_offset_hist": dict(
                sorted(
                    {
                        str(offset): sum(
                            1
                            for record in periodic_candidates
                            if record.offset == int(offset)
                        )
                        for offset in MAIN_OFFSETS
                    }.items(),
                    key=lambda item: int(item[0]),
                )
            ),
            "candidate_count_hist": _candidate_count_hist(
                [record.candidate_bins.size for record in periodic_anchors]
            ),
            "chance_top1_mean": (
                float(
                    np.mean(
                        [1.0 / record.candidate_bins.size for record in periodic_anchors]
                    )
                )
                if periodic_anchors
                else None
            ),
        },
        "retrieval": {
            "k": int(cfg.retrieval_k),
            "anchors": len(retrieval_anchors),
            "evaluation_split_anchor_total": eval_anchor_total,
            "kept_ratio": (
                float(len(retrieval_anchors) / eval_anchor_total)
                if eval_anchor_total
                else None
            ),
            "chance_top1": 1.0 / int(cfg.retrieval_k),
            "candidate_count_hist": {str(cfg.retrieval_k): len(retrieval_anchors)},
            "valid_candidates_hist": dict(
                sorted(
                    per_split_stats[cfg.evaluation_split][
                        "retrieval_valid_candidates_hist"
                    ].items(),
                    key=lambda item: int(item[0]),
                )
            ),
        },
        "per_split": per_split_stats,
    }
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": now_utc_iso(),
        "tokens_version": cfg.tokens_version,
        "audio_version": data_info.get("audio_version"),
        "visual_encoder": cfg.visual_encoder,
        "seed": int(cfg.seed),
        "near_dup_cosine": float(cfg.near_dup),
        "main_k": int(cfg.main_k),
        "retrieval_k": int(cfg.retrieval_k),
        "offsets": list(MAIN_OFFSETS),
        "bin_seconds": BIN_SECONDS,
        "split": {
            name: {"count": len(split[name]), "work_ids": [int(v) for v in split[name]]}
            for name in ("train", "val", "test")
        },
        "evaluation_split": cfg.evaluation_split,
        "data": dict(data_info),
        "candidate_distribution": distribution,
    }
    return ProtocolBundle(
        pairs=dict(pairs),
        split={name: tuple(int(v) for v in split[name]) for name in ("train", "val", "test")},
        main_anchors=main_anchors,
        periodic_anchors=periodic_anchors,
        periodic_candidates=periodic_candidates,
        retrieval_anchors=retrieval_anchors,
        manifest=manifest,
        data_info=dict(data_info),
    )


def _candidate_count_hist(counts: Sequence[int]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for value in counts:
        key = str(int(value))
        hist[key] = int(hist.get(key, 0)) + 1
    return dict(sorted(hist.items(), key=lambda item: int(item[0])))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"不可序列化类型: {type(value).__name__}")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
            stream.write("\n")
        os.replace(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)
    return target


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    tmp_path = Path(tmp_name)
    fieldnames = [
        "model",
        "kind",
        "variant",
        "k",
        "chance_top1",
        "top1",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "n_anchors",
        "n_works",
        "ci95_low",
        "ci95_high",
        "note",
    ]
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        os.replace(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)
    return target


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".tmp.npz", dir=str(target.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)
    return target


def _flatten_records(records: Sequence[AnchorRecord]) -> dict[str, np.ndarray]:
    if not records:
        return {
            "work_id": np.zeros((0,), dtype=np.int64),
            "split_code": np.zeros((0,), dtype=np.int8),
            "anchor_bin": np.zeros((0,), dtype=np.int64),
            "anchor_time": np.zeros((0,), dtype=np.float32),
            "label": np.zeros((0,), dtype=np.int64),
            "available_negatives": np.zeros((0,), dtype=np.int64),
            "candidate_count": np.zeros((0,), dtype=np.int64),
            "candidate_offsets_flat": np.zeros((0,), dtype=np.int64),
            "candidate_bins_flat": np.zeros((0,), dtype=np.int64),
            "candidate_ptr": np.zeros((1,), dtype=np.int64),
        }
    pointers = [0]
    offsets_flat: list[int] = []
    bins_flat: list[int] = []
    for record in records:
        offsets_flat.extend(int(value) for value in record.candidate_offsets)
        bins_flat.extend(int(value) for value in record.candidate_bins)
        pointers.append(len(offsets_flat))
    return {
        "work_id": np.asarray([record.work_id for record in records], dtype=np.int64),
        "split_code": np.asarray(
            [SPLIT_CODES[record.split] for record in records], dtype=np.int8
        ),
        "anchor_bin": np.asarray([record.anchor_bin for record in records], dtype=np.int64),
        "anchor_time": np.asarray(
            [record.anchor_time for record in records], dtype=np.float32
        ),
        "label": np.asarray([record.label for record in records], dtype=np.int64),
        "available_negatives": np.asarray(
            [record.available_negatives for record in records], dtype=np.int64
        ),
        "candidate_count": np.asarray(
            [record.candidate_bins.size for record in records], dtype=np.int64
        ),
        "candidate_offsets_flat": np.asarray(offsets_flat, dtype=np.int64),
        "candidate_bins_flat": np.asarray(bins_flat, dtype=np.int64),
        "candidate_ptr": np.asarray(pointers, dtype=np.int64),
    }




def _flatten_periodic_candidates(
    records: Sequence[PeriodicCandidateRecord],
) -> dict[str, np.ndarray]:
    if not records:
        return {
            "work_id": np.zeros((0,), dtype=np.int64),
            "split_code": np.zeros((0,), dtype=np.int8),
            "anchor_bin": np.zeros((0,), dtype=np.int64),
            "candidate_bin": np.zeros((0,), dtype=np.int64),
            "offset": np.zeros((0,), dtype=np.int64),
        }
    return {
        "work_id": np.asarray([record.work_id for record in records], dtype=np.int64),
        "split_code": np.asarray(
            [SPLIT_CODES[record.split] for record in records], dtype=np.int8
        ),
        "anchor_bin": np.asarray(
            [record.anchor_bin for record in records], dtype=np.int64
        ),
        "candidate_bin": np.asarray(
            [record.candidate_bin for record in records], dtype=np.int64
        ),
        "offset": np.asarray([record.offset for record in records], dtype=np.int64),
    }


def save_protocol(bundle: ProtocolBundle, config: Config) -> dict[str, Any]:
    """把协议 anchor 记录/清单原子写入 data/v05_protocol/<version>/。"""

    root = config.data_path(PROTOCOL_DIRNAME) / bundle.manifest["protocol_version"]
    root.mkdir(parents=True, exist_ok=True)
    main_records = [record for split in ("train", "val", "test") for record in bundle.main_anchors[split]]
    retrieval_records = bundle.retrieval_anchors
    arrays: dict[str, np.ndarray] = {}
    for prefix, records in (("main", main_records), ("periodic", bundle.periodic_anchors)):
        flattened = _flatten_records(records)
        for key, value in flattened.items():
            arrays[f"{prefix}_{key}"] = value
    periodic_candidate_arrays = _flatten_periodic_candidates(bundle.periodic_candidates)
    for key, value in periodic_candidate_arrays.items():
        arrays[f"periodic_candidate_{key}"] = value
    retrieval_arrays = _flatten_records(list(retrieval_records))
    for key, value in retrieval_arrays.items():
        arrays[f"retrieval_{key}"] = value
    anchors_path = _atomic_save_npz(root / ANCHOR_FILENAME, arrays)
    digest = hashlib.sha256(anchors_path.read_bytes()).hexdigest()
    manifest = dict(bundle.manifest)
    manifest.update(
        {
            "anchors_file": config.data_relpath(anchors_path),
            "anchors_sha256": digest,
            "anchors_counts": {
                "main": len(main_records),
                "periodic_anchors": len(bundle.periodic_anchors),
                "periodic_candidates": len(bundle.periodic_candidates),
                "retrieval": len(retrieval_records),
            },
            "saved_at": now_utc_iso(),
        }
    )
    manifest_path = _atomic_write_json(root / MANIFEST_FILENAME, manifest)
    bundle.manifest = manifest
    return {
        "root": str(root),
        "manifest_path": str(manifest_path),
        "anchors_path": str(anchors_path),
        "anchors_sha256": digest,
    }


# ---------------------------------------------------------------- CandidateSet 转换 / 指标


def candidate_sets_from_records(
    pairs: Mapping[int, v05_beta.WorkPair],
    records: Sequence[AnchorRecord],
) -> list[v05_beta.CandidateSet]:
    """把 anchor 记录还原为 v05_beta.CandidateSet（复用 simple_sync/model 评估）。"""

    grouped: dict[int, list[AnchorRecord]] = {}
    for record in records:
        grouped.setdefault(int(record.work_id), []).append(record)
    sets: list[v05_beta.CandidateSet] = []
    for work_id in sorted(grouped):
        pair = pairs.get(work_id)
        if pair is None:
            raise V05ProtocolError(f"缺少 work_id={work_id} 的配对序列")
        items = grouped[work_id]
        anchor_bins = np.asarray([record.anchor_bin for record in items], dtype=np.int64)
        candidates = tuple(
            pair.audio[record.candidate_bins].astype(np.float32) for record in items
        )
        visual_change = None
        candidate_audio_change = None
        candidate_audio_valid = None
        anchor_valid = None
        if pair.visual_change is not None and pair.audio_change is not None and pair.event_valid is not None:
            visual_change = pair.visual_change[anchor_bins].astype(np.float32)
            candidate_audio_change = tuple(
                pair.audio_change[record.candidate_bins].astype(np.float32)
                for record in items
            )
            candidate_audio_valid = tuple(
                pair.event_valid[record.candidate_bins].astype(bool) for record in items
            )
            anchor_valid = pair.event_valid[anchor_bins].astype(bool)
        sets.append(
            v05_beta.CandidateSet(
                work_id=work_id,
                visual=pair.visual[anchor_bins].astype(np.float32),
                candidates=candidates,
                labels=np.asarray([record.label for record in items], dtype=np.int64),
                offsets=tuple(record.candidate_offsets.astype(np.int64) for record in items),
                anchor_times=np.asarray(
                    [record.anchor_time for record in items], dtype=np.float64
                ),
                visual_change=visual_change,
                candidate_audio_change=candidate_audio_change,
                candidate_audio_valid=candidate_audio_valid,
                anchor_valid=anchor_valid,
            )
        )
    return sets


def candidate_sets_from_periodic_candidates(
    pairs: Mapping[int, v05_beta.WorkPair],
    records: Sequence[PeriodicCandidateRecord],
) -> list[v05_beta.CandidateSet]:
    """把近重复候选构造成二元候选集（正样本 vs 该 periodic 候选），供独立评估。"""

    grouped: dict[int, list[PeriodicCandidateRecord]] = {}
    for record in records:
        grouped.setdefault(int(record.work_id), []).append(record)
    sets: list[v05_beta.CandidateSet] = []
    for work_id in sorted(grouped):
        pair = pairs.get(work_id)
        if pair is None:
            raise V05ProtocolError(f"缺少 work_id={work_id} 的配对序列")
        items = grouped[work_id]
        anchor_bins = np.asarray([record.anchor_bin for record in items], dtype=np.int64)
        candidate_bins = np.stack(
            [
                np.asarray([record.anchor_bin, record.candidate_bin], dtype=np.int64)
                for record in items
            ],
            axis=0,
        )
        candidates = tuple(
            pair.audio[bins].astype(np.float32) for bins in candidate_bins
        )
        visual_change = None
        candidate_audio_change = None
        candidate_audio_valid = None
        anchor_valid = None
        if pair.visual_change is not None and pair.audio_change is not None and pair.event_valid is not None:
            visual_change = pair.visual_change[anchor_bins].astype(np.float32)
            candidate_audio_change = tuple(
                pair.audio_change[bins].astype(np.float32) for bins in candidate_bins
            )
            candidate_audio_valid = tuple(
                pair.event_valid[bins].astype(bool) for bins in candidate_bins
            )
            anchor_valid = pair.event_valid[anchor_bins].astype(bool)
        sets.append(
            v05_beta.CandidateSet(
                work_id=work_id,
                visual=pair.visual[anchor_bins].astype(np.float32),
                candidates=candidates,
                labels=np.zeros(len(items), dtype=np.int64),
                offsets=tuple(
                    np.asarray(
                        [0, int(record.candidate_bin - record.anchor_bin)], dtype=np.int64
                    )
                    for record in items
                ),
                anchor_times=np.asarray(
                    [float(pair.times[record.anchor_bin]) for record in items],
                    dtype=np.float64,
                ),
                visual_change=visual_change,
                candidate_audio_change=candidate_audio_change,
                candidate_audio_valid=candidate_audio_valid,
                anchor_valid=anchor_valid,
            )
        )
    return sets


def sample_periodic_candidates(
    records: Sequence[PeriodicCandidateRecord],
    *,
    limit: int = PERIODIC_CANDIDATE_EVAL_LIMIT,
    seed: int = SPLIT_SEED,
) -> list[PeriodicCandidateRecord]:
    """periodic 候选评估的确定性抽样（records 全量仍写入 manifest/npz）。

    近重复候选可能很多（每个 anchor 最多 10 个），评估用固定上限的确定性样本；
    manifest 记录样本上限，metrics 标注抽样，保证所有模型/基线看到同一批候选。
    """

    if len(records) <= int(limit):
        return list(records)
    rng = np.random.default_rng([int(seed), 20251026, 17])
    indices = np.sort(rng.choice(len(records), size=int(limit), replace=False))
    return [records[int(index)] for index in indices]


def _rank_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    k_values: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    """由 (N,K) 分数矩阵计算 Top-1 / Recall@k（逐 anchor 有效候选数行级统计）。"""

    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    mask_array = np.asarray(candidate_mask, dtype=bool)
    if score_array.ndim != 2 or score_array.shape[0] != label_array.size:
        raise V05ProtocolError("scores/labels 形状非法")
    n_anchors = int(label_array.size)
    if n_anchors == 0:
        return {
            "top1": None,
            "recall": {},
            "n_anchors": 0,
            "mean_candidates": None,
        }
    row_indices = np.arange(n_anchors)
    positive_scores = score_array[row_indices, label_array]
    predicted = score_array.argmax(axis=1)
    top1 = float((predicted == label_array).mean())
    valid_counts = mask_array.sum(axis=1)
    recall: dict[str, float] = {}
    for k in k_values:
        eligible = valid_counts >= int(k)
        if not bool(eligible.any()):
            recall[str(int(k))] = None
            continue
        ranks = (score_array > positive_scores[:, None]).sum(axis=1)
        hits = (ranks < int(k)) & eligible
        recall[str(int(k))] = float(hits.sum() / eligible.sum())
    return {
        "top1": top1,
        "recall": recall,
        "n_anchors": n_anchors,
        "mean_candidates": float(valid_counts.mean()),
    }


def _bootstrap_ci(
    per_work_accuracy: Mapping[str, float], *, seed: int, samples: int
) -> dict[str, Any] | None:
    works = sorted(per_work_accuracy)
    if not works:
        return None
    values = np.asarray([float(per_work_accuracy[work]) for work in works], dtype=np.float64)
    mean = float(values.mean())
    if int(samples) <= 0:
        return {"mean": mean, "ci95": [mean, mean], "n_works": len(works)}
    rng = np.random.default_rng([int(seed), 20251026])
    estimates: list[float] = []
    for _ in range(int(samples)):
        picks = rng.integers(0, values.size, size=values.size)
        estimates.append(float(values[picks].mean()))
    low, high = np.percentile(np.asarray(estimates), [2.5, 97.5])
    return {"mean": mean, "ci95": [float(low), float(high)], "n_works": len(works)}


def evaluate_model_on_sets(
    model: Any,
    sets: Sequence[v05_beta.CandidateSet],
    *,
    device: Any,
    batch_size: int,
    k_values: Sequence[int] = (1, 5, 10),
    bootstrap: int = 0,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    """在候选集上评估模型；逐 batch 统计以支持 periodic 集的可变候选数。"""

    if not sets:
        return {
            "top1": None,
            "recall": {},
            "n_anchors": 0,
            "n_works": 0,
            "mean_candidates": None,
            "per_work_accuracy": {},
            "bootstrap": None,
        }
    torch = v05_beta._torch()
    model.eval()
    picks = [
        (set_index, anchor)
        for set_index, item in enumerate(sets)
        for anchor in range(item.n_anchors)
    ]
    if not picks:
        return {
            "top1": None,
            "recall": {},
            "n_anchors": 0,
            "n_works": 0,
            "mean_candidates": None,
            "per_work_accuracy": {},
            "bootstrap": None,
        }
    correct = 0
    total = 0
    candidate_total = 0
    recall_hits = {int(k): 0 for k in k_values}
    recall_totals = {int(k): 0 for k in k_values}
    per_work: dict[int, list[int]] = {}
    with torch.no_grad():
        for start in range(0, len(picks), int(batch_size)):
            batch = v05_beta.collate_anchors(sets, picks[start : start + int(batch_size)])
            batch = {key: value.to(device) for key, value in batch.items()}
            visual_embedding, audio_embedding = model(batch["visual"], batch["audio"])
            scores = torch.einsum("bd,bkd->bk", visual_embedding, audio_embedding)
            scores = scores.masked_fill(~batch["candidate_mask"], -1e9)
            score_array = scores.detach().cpu().numpy()
            labels = batch["labels"].detach().cpu().numpy()
            mask = batch["candidate_mask"].detach().cpu().numpy()
            work_ids = batch["work_ids"].detach().cpu().numpy()
            row_indices = np.arange(labels.size)
            positives = score_array[row_indices, labels]
            predictions = score_array.argmax(axis=1)
            candidate_counts = mask.sum(axis=1)
            candidate_total += int(candidate_counts.sum())
            correct += int((predictions == labels).sum())
            total += int(labels.size)
            ranks = (score_array > positives[:, None]).sum(axis=1)
            for k in k_values:
                eligible = candidate_counts >= int(k)
                recall_totals[int(k)] += int(eligible.sum())
                recall_hits[int(k)] += int(((ranks < int(k)) & eligible).sum())
            for row, work_id in enumerate(work_ids):
                bucket = per_work.setdefault(int(work_id), [0, 0])
                bucket[0] += int(predictions[row] == labels[row])
                bucket[1] += 1
    work_accuracy = {
        str(work_id): float(values[0] / values[1])
        for work_id, values in per_work.items()
        if values[1]
    }
    recall: dict[str, float | None] = {}
    for k in k_values:
        recall[str(int(k))] = (
            float(recall_hits[int(k)] / recall_totals[int(k)])
            if recall_totals[int(k)]
            else None
        )
    metrics = {
        "top1": float(correct / total) if total else None,
        "recall": recall,
        "n_anchors": total,
        "n_works": len(work_accuracy),
        "mean_candidates": (float(candidate_total / total) if total else None),
        "per_work_accuracy": work_accuracy,
        "bootstrap": (
            _bootstrap_ci(work_accuracy, seed=seed, samples=bootstrap)
            if bootstrap > 0
            else None
        ),
    }
    return metrics


def _random_baseline(k: int, *, n_anchors: int, n_works: int | None = None) -> dict[str, Any]:
    chance = 1.0 / int(k)
    return {
        "top1": chance,
        "recall": {str(value): float(min(value, k) / k) for value in (1, 5, 10)},
        "n_anchors": int(n_anchors),
        "n_works": n_works,
        "mean_candidates": float(k),
        "per_work_accuracy": None,
        "bootstrap": None,
        "note": "随机选择候选的理论期望（Top-1=1/K；K<=5 时 Recall@5 无信息量）",
    }


def _simple_sync_score_matrix(
    sets: Sequence[v05_beta.CandidateSet],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """协议 v2 本地 simple_sync 评分（与 v05_beta 定义一致，但输出完整分数矩阵以支持 Recall@10）。"""

    if not sets:
        return (
            np.zeros((0, 0)),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 0), dtype=bool),
            np.zeros((0,), dtype=np.int64),
        )
    if any(
        item.visual_change is None
        or item.candidate_audio_change is None
        or item.candidate_audio_valid is None
        or item.anchor_valid is None
        for item in sets
    ):
        raise V05ProtocolError("simple_sync 需要事件变化标量（visual_change/audio_change）")
    score_rows: list[np.ndarray] = []
    label_rows: list[int] = []
    mask_rows: list[np.ndarray] = []
    work_rows: list[int] = []
    for item in sets:
        visual_change = np.asarray(item.visual_change, dtype=np.float64)
        anchor_valid = np.asarray(item.anchor_valid, dtype=bool)
        candidate_changes = [
            np.asarray(array, dtype=np.float64) for array in item.candidate_audio_change or ()
        ]
        candidate_valid = [
            np.asarray(array, dtype=bool) for array in item.candidate_audio_valid or ()
        ]
        if len(candidate_changes) != item.n_anchors:
            raise V05ProtocolError("simple_sync 候选变化数组与 anchor 数不一致")
        pooled_values: list[np.ndarray] = []
        for values, valid in zip(candidate_changes, candidate_valid, strict=True):
            selected = values[valid]
            if selected.size:
                pooled_values.append(selected)
        if not pooled_values:
            continue
        pooled = np.concatenate(pooled_values)
        if pooled.size < 2 or float(pooled.std()) <= 1e-12:
            continue
        z_values, _ = v05_beta._zscore_with_valid(visual_change, anchor_valid)
        audio_mean = float(pooled.mean())
        audio_std = float(pooled.std())
        for anchor in range(item.n_anchors):
            if not bool(anchor_valid[anchor]):
                continue
            values = candidate_changes[anchor]
            valid = candidate_valid[anchor]
            label = int(item.labels[anchor])
            if label < 0 or label >= values.size or not bool(valid[label]):
                continue
            scores = np.full(values.size, -np.inf, dtype=np.float64)
            z_audio = (values - audio_mean) / audio_std
            scores[valid] = -np.abs(z_values[anchor] - z_audio[valid])
            if not np.isfinite(scores).any():
                continue
            score_rows.append(scores)
            label_rows.append(label)
            mask_rows.append(valid)
            work_rows.append(int(item.work_id))
    max_k = max((row.size for row in score_rows), default=0)
    if not score_rows:
        return (
            np.zeros((0, 0)),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 0), dtype=bool),
            np.zeros((0,), dtype=np.int64),
        )
    scores_matrix = np.full((len(score_rows), max_k), -np.inf, dtype=np.float64)
    masks_matrix = np.zeros((len(score_rows), max_k), dtype=bool)
    for row, (scores, mask) in enumerate(zip(score_rows, mask_rows, strict=True)):
        scores_matrix[row, : scores.size] = scores
        masks_matrix[row, : mask.size] = mask
    return (
        scores_matrix,
        np.asarray(label_rows, dtype=np.int64),
        masks_matrix,
        np.asarray(work_rows, dtype=np.int64),
    )


def _simple_sync_metrics(
    sets: Sequence[v05_beta.CandidateSet], *, bootstrap: int = 0, seed: int = SPLIT_SEED
) -> dict[str, Any]:
    if not sets:
        return {
            "top1": None,
            "recall": {},
            "n_anchors": 0,
            "n_works": 0,
            "per_work_accuracy": None,
            "bootstrap": None,
            "note": "simple_sync 输入为空",
        }
    scores, labels, masks, work_ids = _simple_sync_score_matrix(sets)
    metrics = _rank_metrics(scores, labels, masks, k_values=(1, 5, 10))
    predictions = scores.argmax(axis=1) if scores.size else np.zeros((0,), dtype=np.int64)
    count = int(labels.size)
    per_work: dict[int, list[int]] = {}
    for row in range(count):
        bucket = per_work.setdefault(int(work_ids[row]), [0, 0])
        bucket[0] += int(predictions[row] == labels[row])
        bucket[1] += 1
    work_accuracy = {
        str(work_id): float(values[0] / values[1])
        for work_id, values in per_work.items()
        if values[1]
    }
    metrics["n_works"] = len(work_accuracy)
    metrics["per_work_accuracy"] = work_accuracy
    metrics["bootstrap"] = (
        _bootstrap_ci(work_accuracy, seed=seed, samples=bootstrap)
        if bootstrap > 0
        else None
    )
    metrics["note"] = (
        "simple_sync：0.5s bin 事件统计的视觉变化 vs 音频起始 z-score 负绝对差；"
        "确定性、无学习、无模型权重/作品 ID"
    )
    return metrics


# ---------------------------------------------------------------- 旧 checkpoint 适配


def load_old_checkpoint(
    path: str | Path,
    *,
    protocol_info: Mapping[str, Any],
    device: Any,
) -> LoadedCheckpoint:
    """读取旧 v05_beta checkpoint（model_config/standardizer/state_dict）并还原冻结模型。"""

    import torch

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise V05ProtocolError(f"checkpoint 不存在: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise V05ProtocolError(f"读取 checkpoint 失败（{checkpoint_path}）: {exc}") from exc
    dataset = checkpoint.get("dataset") or {}
    expected_encoder = protocol_info.get("visual_encoder")
    if dataset.get("visual_encoder") not in (None, expected_encoder):
        raise V05ProtocolError(
            f"checkpoint {checkpoint_path.name} 的 visual_encoder="
            f"{dataset.get('visual_encoder')!r} 与协议 {expected_encoder!r} 不一致"
        )
    expected_audio = protocol_info.get("audio_version")
    if dataset.get("audio_version") not in (None, expected_audio):
        raise V05ProtocolError(
            f"checkpoint {checkpoint_path.name} 的 audio_version 与协议不一致"
        )
    model_config = checkpoint.get("model_config") or {}
    standardizer = checkpoint.get("standardizer") or {}
    for key in ("visual_dim", "audio_dim", "hidden", "embed_dim"):
        if key not in model_config:
            raise V05ProtocolError(f"checkpoint 缺少 model_config[{key}]")
    for key in ("visual_mean", "visual_std", "audio_mean", "audio_std"):
        if key not in standardizer:
            raise V05ProtocolError(f"checkpoint 缺少 standardizer[{key}]")
    if int(model_config["visual_dim"]) != int(protocol_info["visual_dim"]):
        raise V05ProtocolError(
            f"checkpoint visual_dim={model_config['visual_dim']} 与协议 "
            f"{protocol_info['visual_dim']} 不一致"
        )
    if int(model_config["audio_dim"]) != int(protocol_info["audio_dim"]):
        raise V05ProtocolError(
            f"checkpoint audio_dim={model_config['audio_dim']} 与协议 "
            f"{protocol_info['audio_dim']} 不一致"
        )
    model = v05_beta.build_model(
        visual_dim=int(model_config["visual_dim"]),
        audio_dim=int(model_config["audio_dim"]),
        standardizer=standardizer,
        hidden=int(model_config["hidden"]),
        embed_dim=int(model_config["embed_dim"]),
    )
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except (KeyError, RuntimeError) as exc:
        raise V05ProtocolError(
            f"checkpoint state_dict 与模型结构不匹配（{checkpoint_path}）: {exc}"
        ) from exc
    model.eval()
    model.to(device)
    info = {
        "name": checkpoint_path.parent.name,
        "path": str(checkpoint_path),
        "seed": checkpoint.get("seed"),
        "model_config": model_config,
        "standardizer": standardizer,
        "checkpoint_dataset": {
            "dataset_version": dataset.get("dataset_version"),
            "audio_version": dataset.get("audio_version"),
            "visual_encoder": dataset.get("visual_encoder"),
            "n_works_available": dataset.get("n_works_available"),
        },
        "old_protocol_metrics": {
            "test_top1": checkpoint.get("test_top1"),
            "best_val_top1": checkpoint.get("best_val_top1"),
        },
    }
    return LoadedCheckpoint(
        name=str(info["name"]), path=checkpoint_path, model=model, info=info
    )


def _build_frozen_model(
    config: Config,
    bundle: ProtocolBundle,
    *,
    checkpoint_standardizer: Mapping[str, np.ndarray] | None,
    device: Any,
    seed: int = 20251026,
) -> Any:
    import torch

    standardizer = checkpoint_standardizer
    if standardizer is None:
        standardizer = v05_beta.fit_standardizer(
            [bundle.pairs[work_id] for work_id in bundle.split["train"]]
        )
    model_config = bundle.data_info.get("model_config") or {}
    torch.manual_seed(int(seed))
    model = v05_beta.build_model(
        visual_dim=int(bundle.data_info["visual_dim"]),
        audio_dim=int(bundle.data_info["audio_dim"]),
        standardizer=standardizer,
        hidden=int(model_config.get("hidden", 128)),
        embed_dim=int(model_config.get("embed_dim", 128)),
    )
    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------- 协议构建入口 / 评估入口


def resolve_device(device: str) -> Any:
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_protocol(
    config: Config,
    cfg: ProtocolConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> ProtocolBundle:
    """真实数据：读取 beta split + 全量 token + audio_dense，构建协议 v2 bundle。"""

    cfg = cfg.validate()
    split = load_beta_split(config)
    beta_audio_version = load_beta_audio_version(config)
    audio_version, audio_dir, audio_manifest = _resolve_audio_version(
        config, cfg.audio_version, beta_audio_version
    )
    work_ids = [work_id for name in ("train", "val", "test") for work_id in split[name]]
    if progress:
        progress(
            f"协议 v2：split {len(split['train'])}/{len(split['val'])}/{len(split['test'])}，"
            f"audio={audio_version}"
        )
    pairs, failed = build_pairs(
        config,
        work_ids,
        tokens_version=cfg.tokens_version,
        audio_version=audio_version,
        visual_encoder=cfg.visual_encoder,
        audio_manifest=audio_manifest,
        progress=progress,
    )
    missing = [work_id for work_id in work_ids if work_id not in pairs]
    if missing:
        raise V05ProtocolError(
            f"协议 v2 缺少 {len(missing)} 部作品的配对序列: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )
    sample_pair = next(iter(pairs.values()))
    data_info = {
        "tokens_version": cfg.tokens_version,
        "audio_version": audio_version,
        "audio_dir": str(audio_dir),
        "visual_encoder": cfg.visual_encoder,
        "visual_dim": int(sample_pair.visual_dim),
        "audio_dim": int(sample_pair.audio_dim),
        "bin_seconds": BIN_SECONDS,
        "n_works": len(pairs),
        "failed_works": failed,
        "beta_split_source": str(config.paths.logs / "v05_beta_summary.json"),
        "model_config": {"hidden": 128, "embed_dim": 128, "temperature": 0.07},
        "notes": [
            "协议 v2 固定 74/21/12 beta split，不新增/删除 works；真实 audio 全量 110 部但协议与 beta 可比使用 107 部",
            "负样本双向偏移；固定 K=6；近重复候选排除出主候选池并进入 periodic 候选集单独评估",
            "不足 5 个非近重复负样本的 anchor 进入 periodic anchor 集；periodic 候选评估使用确定性抽样上限",
            "K=32 retrieval 在 test split 上单独评估；旧 Recall@5 (K<=6) 不作为证据",
        ],
    }
    bundle = build_protocol_bundle(
        pairs, split, config=cfg, data_info=data_info
    )
    bundle.data_info = data_info
    return bundle


def evaluate_protocol(
    config: Config,
    bundle: ProtocolBundle,
    cfg: ProtocolConfig,
    *,
    save_manifest: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """在同一固定协议上评估旧 checkpoints + random/frozen/simple_sync，写 metrics/CSV。"""

    cfg = cfg.validate()
    device = resolve_device(cfg.device)
    if progress:
        progress(f"评估设备: {device}")
    saved = save_protocol(bundle, config) if save_manifest else None
    manifest_path = (
        saved["manifest_path"]
        if saved
        else str(config.data_path(PROTOCOL_DIRNAME) / cfg.version / MANIFEST_FILENAME)
    )
    eval_split = cfg.evaluation_split
    main_sets = candidate_sets_from_records(bundle.pairs, bundle.main_anchors[eval_split])
    periodic_sets = candidate_sets_from_records(
        bundle.pairs,
        [record for record in bundle.periodic_anchors if record.split == eval_split],
    )
    retrieval_sets = candidate_sets_from_records(
        bundle.pairs,
        [
            AnchorRecord(
                work_id=record.work_id,
                anchor_bin=record.anchor_bin,
                anchor_time=record.anchor_time,
                split=record.split,
                group="retrieval",
                label=record.label,
                candidate_bins=record.candidate_bins,
                candidate_offsets=record.candidate_offsets,
                available_negatives=record.candidate_bins.size - 1,
            )
            for record in bundle.retrieval_anchors
            if record.split == eval_split
        ],
    )
    periodic_candidate_eval_records = sample_periodic_candidates(
        [
            record
            for record in bundle.periodic_candidates
            if record.split == eval_split
        ],
        limit=PERIODIC_CANDIDATE_EVAL_LIMIT,
        seed=cfg.seed,
    )
    periodic_candidate_sets = candidate_sets_from_periodic_candidates(
        bundle.pairs, periodic_candidate_eval_records
    )
    if not main_sets:
        raise V05ProtocolError("主协议候选集为空")
    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}

    def add_result(
        name: str,
        kind: str,
        variant: str,
        k: int,
        metrics: Mapping[str, Any],
        chance_top1: float | None,
        note: str = "",
    ) -> None:
        bootstrap = metrics.get("bootstrap")
        ci = bootstrap.get("ci95") if isinstance(bootstrap, Mapping) else None
        row = {
            "model": name,
            "kind": kind,
            "variant": variant,
            "k": int(k),
            "chance_top1": chance_top1,
            "top1": metrics.get("top1"),
            "recall_at_1": (metrics.get("recall") or {}).get("1"),
            "recall_at_5": (metrics.get("recall") or {}).get("5"),
            "recall_at_10": (metrics.get("recall") or {}).get("10"),
            "n_anchors": metrics.get("n_anchors"),
            "n_works": metrics.get("n_works"),
            "ci95_low": ci[0] if ci else None,
            "ci95_high": ci[1] if ci else None,
            "note": note,
        }
        rows.append(row)
        results.setdefault(name, {"kind": kind, "variants": {}})["variants"][variant] = {
            **row,
            "per_work_accuracy": metrics.get("per_work_accuracy"),
            "bootstrap": bootstrap,
        }

    random_main = _random_baseline(cfg.main_k, n_anchors=sum(item.n_anchors for item in main_sets), n_works=len(main_sets))
    add_result(
        "random",
        "baseline",
        "main_k6",
        cfg.main_k,
        random_main,
        1.0 / cfg.main_k,
        "理论随机基线；K=6 固定后 random Top1=1/6",
    )
    random_retrieval = _random_baseline(
        cfg.retrieval_k,
        n_anchors=sum(item.n_anchors for item in retrieval_sets),
        n_works=len(retrieval_sets),
    )
    add_result(
        "random",
        "baseline",
        "retrieval_k32",
        cfg.retrieval_k,
        random_retrieval,
        1.0 / cfg.retrieval_k,
        "理论随机：Top1=3.125%，Recall@5=15.625%，Recall@10=31.25%",
    )
    periodic_chance = (
        float(
            np.mean(
                [
                    1.0 / len(item.candidates[anchor])
                    for item in periodic_sets
                    for anchor in range(item.n_anchors)
                ]
            )
        )
        if periodic_sets
        else None
    )
    add_result(
        "random",
        "baseline",
        "periodic",
        cfg.main_k,
        {
            "top1": periodic_chance,
            "recall": {"1": periodic_chance},
            "n_anchors": sum(item.n_anchors for item in periodic_sets),
            "n_works": len(periodic_sets),
            "per_work_accuracy": None,
            "bootstrap": None,
        },
        periodic_chance,
        "periodic anchor 集候选数可变，chance=mean(1/K)",
    )
    periodic_candidate_chance = 0.5
    add_result(
        "random",
        "baseline",
        "periodic_candidate",
        2,
        {
            "top1": periodic_candidate_chance if periodic_candidate_sets else None,
            "recall": {"1": periodic_candidate_chance} if periodic_candidate_sets else {},
            "n_anchors": sum(item.n_anchors for item in periodic_candidate_sets),
            "n_works": len(periodic_candidate_sets),
            "per_work_accuracy": None,
            "bootstrap": None,
        },
        periodic_candidate_chance,
        f"近重复候选独立二元集（正样本 vs 1 个 periodic 候选），chance=50%；"
        f"全量候选 {len(bundle.periodic_candidates)}，评估抽样上限 {PERIODIC_CANDIDATE_EVAL_LIMIT}",
    )

    simple_main = _simple_sync_metrics(main_sets, bootstrap=cfg.bootstrap, seed=cfg.seed)
    add_result(
        "simple_sync",
        "baseline",
        "main_k6",
        cfg.main_k,
        simple_main,
        1.0 / cfg.main_k,
        "v05_beta simple_sync：事件统计 z-score 负绝对差（固定协议候选）",
    )
    simple_retrieval = _simple_sync_metrics(
        retrieval_sets, bootstrap=cfg.bootstrap, seed=cfg.seed
    )
    add_result(
        "simple_sync",
        "baseline",
        "retrieval_k32",
        cfg.retrieval_k,
        simple_retrieval,
        1.0 / cfg.retrieval_k,
        "simple_sync on K=32 retrieval 候选池",
    )
    simple_periodic = _simple_sync_metrics(
        periodic_sets, bootstrap=cfg.bootstrap, seed=cfg.seed
    )
    add_result(
        "simple_sync",
        "baseline",
        "periodic",
        cfg.main_k,
        simple_periodic,
        periodic_chance,
        "simple_sync on periodic anchor 集",
    )
    simple_periodic_candidates = _simple_sync_metrics(
        periodic_candidate_sets, bootstrap=cfg.bootstrap, seed=cfg.seed
    )
    add_result(
        "simple_sync",
        "baseline",
        "periodic_candidate",
        2,
        simple_periodic_candidates,
        0.5,
        "simple_sync on periodic 候选二元集",
    )

    # 旧 checkpoints
    checkpoint_paths = [Path(path) for path in cfg.checkpoint_paths]
    if not checkpoint_paths:
        checkpoint_paths = [
            config.data_path("models") / f"v05_beta_{seed}" / "checkpoint.pt"
            for seed in (42, 43, 44)
        ]
    first_standardizer: Mapping[str, np.ndarray] | None = None
    loaded_models: list[LoadedCheckpoint] = []
    for path in checkpoint_paths:
        loaded = load_old_checkpoint(
            path,
            protocol_info={
                **bundle.data_info,
                "visual_encoder": bundle.data_info["visual_encoder"],
                "audio_version": bundle.data_info["audio_version"],
            },
            device=device,
        )
        loaded_models.append(loaded)
        if first_standardizer is None:
            first_standardizer = loaded.info["standardizer"]
    # frozen projection 使用与旧模型相同的标准化统计（若可用），否则 train 拟合
    frozen_model = _build_frozen_model(
        config,
        bundle,
        checkpoint_standardizer=first_standardizer,
        device=device,
    )
    frozen_main = evaluate_model_on_sets(
        frozen_model,
        main_sets,
        device=device,
        batch_size=cfg.batch_size,
        k_values=(1, 5, 10),
        bootstrap=cfg.bootstrap,
        seed=cfg.seed,
    )
    add_result(
        "frozen_projection",
        "baseline",
        "main_k6",
        cfg.main_k,
        frozen_main,
        1.0 / cfg.main_k,
        "同结构随机初始化冻结投影（不训练）",
    )
    frozen_retrieval = evaluate_model_on_sets(
        frozen_model,
        retrieval_sets,
        device=device,
        batch_size=cfg.batch_size,
        k_values=(1, 5, 10),
        bootstrap=cfg.bootstrap,
        seed=cfg.seed,
    )
    add_result(
        "frozen_projection",
        "baseline",
        "retrieval_k32",
        cfg.retrieval_k,
        frozen_retrieval,
        1.0 / cfg.retrieval_k,
        "同结构随机初始化冻结投影（K=32 retrieval）",
    )
    frozen_periodic = evaluate_model_on_sets(
        frozen_model,
        periodic_sets,
        device=device,
        batch_size=cfg.batch_size,
        k_values=(1,),
        bootstrap=cfg.bootstrap,
        seed=cfg.seed,
    )
    add_result(
        "frozen_projection",
        "baseline",
        "periodic",
        cfg.main_k,
        frozen_periodic,
        periodic_chance,
        "冻结投影 on periodic anchor 集",
    )
    frozen_periodic_candidates = evaluate_model_on_sets(
        frozen_model,
        periodic_candidate_sets,
        device=device,
        batch_size=cfg.batch_size,
        k_values=(1,),
        bootstrap=cfg.bootstrap,
        seed=cfg.seed,
    )
    add_result(
        "frozen_projection",
        "baseline",
        "periodic_candidate",
        2,
        frozen_periodic_candidates,
        0.5,
        "冻结投影 on periodic 候选二元集",
    )

    for loaded in loaded_models:
        if progress:
            progress(f"评估旧 checkpoint: {loaded.name}")
        main_metrics = evaluate_model_on_sets(
            loaded.model,
            main_sets,
            device=device,
            batch_size=cfg.batch_size,
            k_values=(1, 5, 10),
            bootstrap=cfg.bootstrap,
            seed=cfg.seed,
        )
        add_result(
            loaded.name,
            "checkpoint",
            "main_k6",
            cfg.main_k,
            main_metrics,
            1.0 / cfg.main_k,
            "旧 v05_beta checkpoint 在固定 K=6 双向偏移协议下重评",
        )
        retrieval_metrics = evaluate_model_on_sets(
            loaded.model,
            retrieval_sets,
            device=device,
            batch_size=cfg.batch_size,
            k_values=(1, 5, 10),
            bootstrap=cfg.bootstrap,
            seed=cfg.seed,
        )
        add_result(
            loaded.name,
            "checkpoint",
            "retrieval_k32",
            cfg.retrieval_k,
            retrieval_metrics,
            1.0 / cfg.retrieval_k,
            "旧 v05_beta checkpoint 在 K=32 retrieval 候选池重评",
        )
        periodic_metrics = evaluate_model_on_sets(
            loaded.model,
            periodic_sets,
            device=device,
            batch_size=cfg.batch_size,
            k_values=(1,),
            bootstrap=cfg.bootstrap,
            seed=cfg.seed,
        )
        add_result(
            loaded.name,
            "checkpoint",
            "periodic",
            cfg.main_k,
            periodic_metrics,
            periodic_chance,
            "旧 v05_beta checkpoint 在 periodic anchor 集重评",
        )
        periodic_candidate_metrics = evaluate_model_on_sets(
            loaded.model,
            periodic_candidate_sets,
            device=device,
            batch_size=cfg.batch_size,
            k_values=(1,),
            bootstrap=cfg.bootstrap,
            seed=cfg.seed,
        )
        add_result(
            loaded.name,
            "checkpoint",
            "periodic_candidate",
            2,
            periodic_candidate_metrics,
            0.5,
            "旧 v05_beta checkpoint 在 periodic 候选二元集重评",
        )
        results[loaded.name]["info"] = loaded.info

    metrics_payload = {
        "protocol_version": cfg.version,
        "created_at": now_utc_iso(),
        "protocol_manifest": manifest_path,
        "data": bundle.data_info,
        "split": {
            "counts": [len(bundle.split[name]) for name in ("train", "val", "test")],
            "evaluation_split": eval_split,
            "work_ids": {
                name: [int(value) for value in bundle.split[name]]
                for name in ("train", "val", "test")
            },
        },
        "candidate_distribution": bundle.manifest.get("candidate_distribution"),
        "candidate_sets": {
            "main": {
                "n_anchors": sum(item.n_anchors for item in main_sets),
                "n_works": len(main_sets),
                "mean_candidates": float(
                    np.mean([len(item.candidates[0]) for item in main_sets])
                ),
            },
            "periodic": {
                "n_anchors": sum(item.n_anchors for item in periodic_sets),
                "n_works": len(periodic_sets),
                "candidate_count_hist": _candidate_count_hist(
                    [len(item.candidates[0]) for item in periodic_sets]
                ),
            },
            "retrieval": {
                "n_anchors": sum(item.n_anchors for item in retrieval_sets),
                "n_works": len(retrieval_sets),
                "mean_candidates": float(
                    np.mean([len(item.candidates[0]) for item in retrieval_sets])
                )
                if retrieval_sets
                else None,
            },
            "periodic_candidate": {
                "n_anchors": sum(item.n_anchors for item in periodic_candidate_sets),
                "n_works": len(periodic_candidate_sets),
                "total_candidates": sum(
                    1
                    for record in bundle.periodic_candidates
                    if record.split == eval_split
                ),
                "evaluation_limit": PERIODIC_CANDIDATE_EVAL_LIMIT,
                "sampled": len(periodic_candidate_eval_records),
                "candidate_count_per_anchor": 2,
            },
        },
        "results": results,
        "rows": rows,
        "notes": [
            "旧 beta Recall@5 在 K<=6 候选下无信息量；本协议只以 K=6 Top-1/Recall@1 与 K=32 Recall@1/5/10 作为证据",
            "旧 checkpoints 的训练协议为可变 K<=6、仅未来偏移、近重复删除；与新协议数字不可直接比较",
            "旧模型没有用新协议训练；本评估只回答“旧模型在新协议下的能力”",
            "近重复候选不进入主 K=6 候选池；不足 5 个非近重复负样本的 anchor 进入 periodic anchor 集，所有被排除的近重复候选进入 periodic candidate 集并分别评估",
            "periodic candidate 评估使用确定性抽样上限（见 candidate_sets.periodic_candidate.evaluation_limit / sampled），所有模型/基线使用同一批候选",
            "不注册 DB、不建 map、不写 data/representations",
        ],
    }
    metrics_path = _atomic_write_json(
        config.paths.logs / "v05_protocol_v2_metrics.json", metrics_payload
    )
    csv_path = _atomic_write_csv(
        config.paths.logs / "v05_protocol_v2_metrics.csv", rows
    )
    metrics_payload["artifacts"] = {
        "protocol_manifest": manifest_path,
        "metrics": str(metrics_path),
        "csv": str(csv_path),
    }
    _atomic_write_json(metrics_path, metrics_payload)
    return metrics_payload
