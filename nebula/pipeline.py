"""流水线：下载 -> 切片 -> 推理 -> 保存 -> 登记 -> 清理。

支持两种采样模式（Pipeline(config, sampling_mode=...) 可覆盖 config.sampling.mode）：

legacy（v0.3，默认）：
1. 解析来源：本地文件直接使用（永不删除）；Bilibili 作品优先复用 sources 持久缓存，
   未命中时先取元数据再下载到 sources（temp 仅作下载暂存）。
2. ffprobe 读取时长与音视频流信息，按 nebula.sampling.plan_clips 生成均匀片段（<=N 段）。
3. ffmpeg 截取音画同步片段（无音轨 mp4 + wav），调用 Embedder 得到片段向量。
4. 逐行归一化 -> 求平均 -> 再归一化得到作品向量，写 data/embeddings 并登记 SQLite。

full_coverage（v0.4）：
1. 同 legacy 的取源与 probe；超长视频（> media.long_video_seconds）默认拒绝，需 --allow-long。
2. plan_full_coverage 覆盖整条时间轴；窗口计划与参数写入 .partial/plan.json。
3. 可选高频事件扫描（nebula.events，延迟 import），产物为 event_scan.npz。
4. 按 model.batch_size 分批：只对缺失批次提取 -> 编码 -> 原子写 batch_{k}.npz ->
   删该批临时切片 -> 更新 sequence_states；中断后按 plan.json 校验并从缺失批继续。
5. 全部完成后拼接 clip_*.npy / clip_times.npy / work_*.npy / manifest.json，
   replace_clips（5 元组）-> register_embedding（六种 kind）-> assembled -> complete，
   再删除 .partial 与 temp。失败保留 .partial/temp 供恢复。

两种模式都用“先写文件再登记 DB”；已有当前版本向量的作品直接跳过，换参数生成新版本，
旧版本文件保留。Bilibili 源视频持久缓存在 data/sources/<bvid>/<bvid>_p<part>.*
（优先复用可 probe 的缓存，不删除缓存；temp 只放切片与下载暂存）。
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import __version__
from .config import Config
from .db import Catalog
from .errors import DatabaseError, EmbeddingError, MediaError, PipelineError
from .media import (
    black_video_clip,
    clip_duration_report,
    extract_clip,
    mute_audio_clip,
    probe_media,
)
from .sampling import ClipPlan, CoverageClip, plan_clips, plan_full_coverage
from .util import now_utc_iso, slugify

logger = logging.getLogger(__name__)


def _load_resolve_bilibili_source() -> Any | None:
    """延迟导入 t3 的 Bilibili 源缓存 helper；尚未交付时返回 None。

    helper 契约（nebula/download.py）::

        resolve_bilibili_source(config, client, bvid, part, temp_dir, download,
                                *, timeout_s=600.0, downloader=None) -> Path

    缓存路径 data/sources/<bvid>/<bvid>_p<part>.*：先 probe 复用，下载到 temp
    暂存后原子移动进 sources，失败清理暂存但不删除已有缓存。
    """

    try:
        from .download import resolve_bilibili_source
    except ImportError:
        return None
    return resolve_bilibili_source


CLIP_SUFFIX = "__f{frames}_clip{clip_seconds}s_x{clips}_{sr}k"
FULL_SUFFIX = "__f{frames}_full{clip_seconds}s_ov{overlap}_{sr}k{height}"

SAMPLING_MODES = ("legacy", "full_coverage")


@dataclass
class WorkOutcome:
    """单部作品的处理结果。"""

    work_id: int
    status: str  # completed | failed | skipped
    message: str | None = None
    n_clips: int = 0
    duration_s: float | None = None
    elapsed_s: float = 0.0


@dataclass
class PipelineStats:
    """一次批处理的汇总。"""

    outcomes: list[WorkOutcome] = field(default_factory=list)
    selected: int = 0

    @property
    def completed(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "completed")

    @property
    def failed(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "failed")

    @property
    def skipped(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "skipped")

    def summary(self) -> str:
        return (
            f"完成 {self.completed} 部，失败 {self.failed} 部，跳过 {self.skipped} 部"
            f"（本次候选 {self.selected} 部）"
        )


class Pipeline:
    """处理流水线。线程不安全；一次运行只用一个实例。"""

    def __init__(
        self,
        config: Config,
        *,
        catalog: Catalog | None = None,
        embedder: Any | None = None,
        allow_test_embedder: bool = False,
        sampling_mode: str | None = None,
        allow_long: bool = False,
    ) -> None:
        mode = sampling_mode if sampling_mode is not None else config.sampling.mode
        if mode not in SAMPLING_MODES:
            raise PipelineError(
                f"未知采样模式: {mode!r}（只支持 {', '.join(SAMPLING_MODES)}）"
            )
        self.sampling_mode = mode
        self.allow_long = bool(allow_long)
        self.config = config
        self.config.ensure_directories()
        self._owns_catalog = catalog is None
        self.catalog = catalog if catalog is not None else Catalog(config.paths.db_path)
        self.catalog.initialize()
        self._embedder = embedder
        self._owns_embedder = embedder is None
        self.allow_test_embedder = allow_test_embedder
        self._version: str | None = None
        self._bilibili_client: Any | None = None
        self._run_id: int | None = None
        self.sync_reports: list[dict[str, Any]] = []
        self._sync_budget = 0

    # ------------------------------------------------------------------ 生命周期
    def close(self) -> None:
        if self._owns_embedder and self._embedder is not None:
            close = getattr(self._embedder, "close", None)
            if callable(close):
                close()
            self._embedder = None
        if self._owns_catalog:
            self.catalog.close()

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    # ------------------------------------------------------------------ 嵌入器与版本
    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            from .embed import PEAVEmbedder

            self._embedder = PEAVEmbedder(self.config.model)
            self._owns_embedder = True
        if getattr(self._embedder, "is_test_only", False) and not self.allow_test_embedder:
            raise PipelineError(
                "当前使用测试用嵌入器（is_test_only），只能显式启用 allow_test_embedder=True 时使用"
            )
        return self._embedder

    @property
    def embedding_version(self) -> str:
        """由嵌入器版本 + 采样参数拼出的序列版本；不同版本绝不覆盖。"""

        if self._version is None:
            if self.sampling_mode == "full_coverage":
                suffix = FULL_SUFFIX.format(
                    frames=self.config.model.frames,
                    clip_seconds=f"{self.config.media.clip_seconds:g}",
                    overlap=f"{self.config.sampling.overlap_seconds:g}",
                    sr=self.config.media.audio_sample_rate // 1000,
                    height=(
                        f"_h{int(self.config.media.clip_height)}"
                        if self.config.media.clip_height
                        else ""
                    ),
                )
            else:
                suffix = CLIP_SUFFIX.format(
                    frames=self.config.model.frames,
                    clip_seconds=f"{self.config.media.clip_seconds:g}",
                    clips=self.config.media.clips_per_work,
                    sr=self.config.media.audio_sample_rate // 1000,
                )
            self._version = slugify(self.embedder.version) + suffix
        return self._version

    def _ensure_version(self) -> str:
        version = self.embedding_version
        embedder = self.embedder
        is_test = bool(getattr(embedder, "is_test_only", False))
        if self.sampling_mode == "full_coverage":
            params = {
                "embedder": type(embedder).__name__,
                "is_test_only": is_test,
                "sampling_mode": "full_coverage",
                "full_coverage": True,
                "overlap_seconds": float(self.config.sampling.overlap_seconds),
                "clip_height": self.config.media.clip_height,
                "aggregation": "l2-normalize-mean-l2-normalize",
                "preprocess": (
                    "ffmpeg full-coverage windows h264/yuv420p"
                    + (
                        f" scale=-2:{int(self.config.media.clip_height)}"
                        if self.config.media.clip_height
                        else " (native resolution)"
                    )
                    + " + wav 48k/2ch/pcm_s16le"
                ),
            }
            clips_per_work = 0
        else:
            params = {
                "embedder": type(embedder).__name__,
                "is_test_only": is_test,
                "aggregation": "l2-normalize-mean-l2-normalize",
                "preprocess": "ffmpeg 8s clip h264/yuv420p + wav 48k/2ch/pcm_s16le",
                "ablation": None,
            }
            clips_per_work = self.config.media.clips_per_work
        self.catalog.register_embedding_version(
            version,
            model_checkpoint=(
                self.config.model.checkpoint if not is_test else f"test-only:{embedder.version}"
            ),
            model_revision=None if is_test else self.config.model.revision,
            frames=self.config.model.frames,
            clip_seconds=self.config.media.clip_seconds,
            clips_per_work=clips_per_work,
            audio_sample_rate=self.config.media.audio_sample_rate,
            params=params,
        )
        return version

    # ------------------------------------------------------------------ 对外流程
    def process_pending(
        self,
        *,
        source: str | None = None,
        limit: int | None = None,
        retry_failed: bool = True,
        verify_sync: int = 0,
        list_key: str | None = None,
        list_limit: int | None = None,
    ) -> PipelineStats:
        """处理尚未完成的作品。

        参数
        ----
        source: None | "local" | "bilibili"
        limit: 最多处理多少部（用于试跑）
        retry_failed: 是否重试 failed 作品
        verify_sync: 对前 N 部作品抽查抽出的音画片段时长是否一致（P0 同步检查）
        list_key: 只处理某个榜单（如 votes_2025）中的作品，按榜单排名排序
        list_limit: 结合 list_key，只取榜单前 N 条（先截取再过滤已完成）
        """

        if source not in (None, "local", "bilibili"):
            raise PipelineError(f"未知来源: {source}")
        if limit is not None and limit < 1:
            raise PipelineError("limit 必须 >= 1")
        if list_limit is not None and list_limit < 1:
            raise PipelineError("list_limit 必须 >= 1")
        version = self._ensure_version()
        works = self._select_works(
            version=version,
            source=source,
            limit=limit,
            retry_failed=retry_failed,
            list_key=list_key,
            list_limit=list_limit,
        )
        embedder = self.embedder
        params = {
            "source": source,
            "limit": limit,
            "retry_failed": retry_failed,
            "verify_sync": verify_sync,
            "list_key": list_key,
            "list_limit": list_limit,
            "embedder": type(embedder).__name__,
            "is_test_only": bool(getattr(embedder, "is_test_only", False)),
        }
        return self._execute_batch(
            works, version=version, params=params, verify_sync=verify_sync
        )

    def retry_works(self, work_ids: Sequence[int]) -> PipelineStats:
        """把指定作品重置为 pending 后立即重新处理（单部/批量重试入口）。

        注意：不经过 process_pending 的 has_embedding 跳过逻辑，因此即使作品
        已有当前版本向量也会重新编码（用于复现性检查或修复坏向量）。
        """

        if not work_ids:
            raise PipelineError("请至少提供一个 work_id")
        version = self._ensure_version()
        works: list[dict[str, Any]] = []
        for work_id in work_ids:
            work = self.catalog.get_work(int(work_id))
            if work is None:
                raise PipelineError(f"作品不存在: {work_id}")
            self.catalog.reopen(int(work_id), reason="按用户要求重试")
            refreshed = self.catalog.get_work(int(work_id))
            assert refreshed is not None
            works.append(refreshed)
        return self._execute_batch(
            works,
            version=version,
            params={"mode": "retry", "work_ids": [int(w) for w in work_ids]},
            verify_sync=0,
        )

    def _execute_batch(
        self,
        works: Sequence[dict[str, Any]],
        *,
        version: str,
        params: Mapping[str, Any],
        verify_sync: int = 0,
    ) -> PipelineStats:
        """执行一个批次：登记 pipeline_runs、逐部处理、写分阶段耗时与汇总。"""

        self._sync_budget = max(0, int(verify_sync))
        self.sync_reports = []
        stats = PipelineStats(selected=len(works))
        run_id = self.catalog.start_pipeline_run(embedding_version=version, params=params)
        self._run_id = run_id
        batch_started = time.monotonic()
        logger.info("批次 run_id=%s 开始：候选 %d 部（向量版本 %s）", run_id, len(works), version)
        try:
            for index, work in enumerate(works, start=1):
                logger.info(
                    "[%d/%d] 处理 work_id=%s (%s)",
                    index,
                    len(works),
                    work["id"],
                    work["source"],
                )
                stats.outcomes.append(
                    self._process_work(work, version=version, run_id=run_id)
                )
        finally:
            total_s = time.monotonic() - batch_started
            model_load_s = (
                float(getattr(self._embedder, "model_load_s", 0.0) or 0.0)
                if self._embedder is not None
                else 0.0
            )
            self.catalog.finish_pipeline_run(
                run_id,
                n_selected=len(works),
                n_completed=stats.completed,
                n_failed=stats.failed,
                total_s=total_s,
                model_load_s=model_load_s,
            )
            perf_path = self._write_perf_csv(run_id)
            self._run_id = None
            logger.info(
                "批次 run_id=%s 结束：完成 %d 失败 %d，总耗时 %.2fs，模型加载 %.2fs，"
                "分阶段耗时 %s",
                run_id,
                stats.completed,
                stats.failed,
                total_s,
                model_load_s,
                perf_path,
            )
        if self.sync_reports:
            self._write_sync_report()
        return stats

    def _write_perf_csv(self, run_id: int) -> Path | None:
        """把当前批次的单部分阶段耗时写成 CSV（A4）；无数据时返回 None。"""

        rows = self.catalog.list_work_timings(run_id=run_id)
        if not rows:
            return None
        path = self.config.paths.logs / f"perf_{run_id:03d}.csv"
        columns = [
            "run_id",
            "work_id",
            "embedding_version",
            "download_s",
            "probe_s",
            "extract_s",
            "preprocess_s",
            "forward_s",
            "save_s",
            "events_s",
            "total_s",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name) for name in columns})
        logger.info("分阶段耗时写入 %s（%d 行）", path, len(rows))
        return path

    def status(self) -> dict[str, Any]:
        stats = self.catalog.stats()
        stats["temp_dirs"] = self._temp_dir_count()
        return stats

    def clean_temp(self) -> int:
        """清空 temp 下所有流水线临时目录（并返回删除的数量）。"""

        removed = 0
        if not self.config.paths.temp.is_dir():
            return 0
        for entry in self.config.paths.temp.iterdir():
            if entry.name == ".gitkeep":
                continue
            try:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                removed += 1
            except OSError as exc:  # pragma: no cover - 文件占用等
                logger.warning("无法删除临时文件 %s: %s", entry, exc)
        return removed

    # ------------------------------------------------------------------ 选择作品
    def _heal_interrupted_work(self, work_id: int, *, version: str, status: str) -> None:
        """修复“向量已登记但状态卡在 downloading/encoding”的崩溃窗口。

        典型场景：进程在向量文件写入并登记之后、catalog.complete() 之前被强杀。
        若片段记录也完整，则直接补一个 completed 迁移（留 event 痕迹）；
        若片段记录缺失，则 reopen 让本次批处理重新编码补全。
        """

        clips = self.catalog.get_clips(work_id, version=version)
        try:
            if clips:
                # downloading -> completed 不是合法迁移；先补一步 downloading -> encoding。
                if status == "downloading":
                    self.catalog.set_status(work_id, "encoding")
                self.catalog.complete(work_id)
                logger.warning(
                    "work_id=%s 的向量已存在但状态残留为 %s，已自愈为 completed",
                    work_id,
                    status,
                )
            else:
                self.catalog.reopen(work_id, reason="向量已登记但片段记录缺失，重新处理")
                logger.warning(
                    "work_id=%s 状态残留为 %s 且片段记录缺失，已重新排队处理",
                    work_id,
                    status,
                )
        except DatabaseError:
            # 自愈失败不能拖垮整批选择流程：保留原状态，交由本次处理流程按常规路径恢复。
            logger.exception(
                "work_id=%s 自愈失败（原状态 %s），本次按常规流程处理", work_id, status
            )

    def _select_works(
        self,
        *,
        version: str,
        source: str | None,
        limit: int | None,
        retry_failed: bool,
        list_key: str | None = None,
        list_limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if list_key is not None:
            candidates = self.catalog.list_works_for_list(list_key)
            if list_limit is not None:
                candidates = candidates[:list_limit]
        else:
            candidates = self.catalog.list_works(source=source)
        duplicates = self.catalog.get_duplicate_work_ids()
        selected: list[dict[str, Any]] = []
        for work in candidates:
            work_id = int(work["id"])
            if work_id in duplicates:
                # 重复实体默认不处理；canonical 已有向量时会被直接跳过
                continue
            if source is not None and work["source"] != source:
                continue
            if self.catalog.has_embedding(work_id, version=version):
                status = str(work["status"])
                if status in ("downloading", "encoding"):
                    self._heal_interrupted_work(work_id, version=version, status=status)
                    if str(self.catalog.get_work(work_id)["status"]) != "completed":
                        selected.append(work)
                        if limit is not None and len(selected) >= limit:
                            break
                continue
            status = str(work["status"])
            if status == "failed" and not retry_failed:
                continue
            if status not in ("pending", "failed", "completed", "downloading", "encoding"):
                continue
            selected.append(work)
            if limit is not None and len(selected) >= limit:
                break
        return selected

    # ------------------------------------------------------------------ 处理单部
    def _process_work(
        self, work: dict[str, Any], *, version: str, run_id: int | None = None
    ) -> WorkOutcome:
        """处理一部作品：按 sampling_mode 走 legacy 或 full_coverage 分支。"""

        work_id = int(work["id"])
        started = time.monotonic()
        temp_dir = self.config.paths.temp / f"work_{work_id:06d}"
        outcome = WorkOutcome(work_id=work_id, status="failed")
        timing: dict[str, float] = {
            "download_s": 0.0,
            "probe_s": 0.0,
            "extract_s": 0.0,
            "preprocess_s": 0.0,
            "forward_s": 0.0,
            "events_s": 0.0,
            "save_s": 0.0,
        }
        try:
            if str(work["status"]) in ("completed", "downloading", "encoding"):
                self.catalog.reopen(
                    work_id, reason=f"为向量版本 {version} 重新处理（旧向量保留）"
                )
            source_path = self._resolve_source(work, work_id, temp_dir, timing)

            probe_started = time.monotonic()
            info = probe_media(
                source_path,
                ffprobe=self.config.media.ffprobe,
                timeout_s=self.config.media.probe_timeout_seconds,
            )
            timing["probe_s"] = time.monotonic() - probe_started
            if not info.has_video:
                raise MediaError(f"作品没有视频流，无法提取音画向量: {source_path}")
            self.catalog.update_work_metadata(
                work_id,
                duration_s=info.duration_s,
                has_audio=int(info.has_audio),
                has_video=int(info.has_video),
            )

            if self.sampling_mode == "full_coverage":
                n_clips = self._run_full_coverage(
                    work_id, source_path, info, version, temp_dir, timing
                )
            else:
                n_clips = self._run_legacy(
                    work_id, source_path, info, version, temp_dir, timing
                )

            self.catalog.complete(work_id)
            outcome = WorkOutcome(
                work_id=work_id,
                status="completed",
                n_clips=n_clips,
                duration_s=round(info.duration_s, 3),
            )
            # 只有向量成功写入并登记后才清理临时文件
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(
                "work_id=%s 完成（%s）：%d 个片段，%.1fs",
                work_id,
                self.sampling_mode,
                n_clips,
                info.duration_s,
            )
        except Exception as exc:  # noqa: BLE001 - 批处理需要捕获任何单部失败
            message = f"{type(exc).__name__}: {exc}"
            logger.exception("work_id=%s 处理失败: %s", work_id, message)
            try:
                self.catalog.fail(work_id, message)
            except DatabaseError:  # pragma: no cover - 编目本身异常
                logger.exception("记录失败状态时出错 work_id=%s", work_id)
            outcome = WorkOutcome(work_id=work_id, status="failed", message=message)
        finally:
            total_s = time.monotonic() - started
            if run_id is not None:
                try:
                    self.catalog.record_work_timing(
                        run_id,
                        work_id,
                        embedding_version=version,
                        total_s=total_s,
                        **timing,
                    )
                except DatabaseError:  # pragma: no cover - 计时记录失败不影响主流程
                    logger.exception("记录分阶段耗时失败 work_id=%s", work_id)
        outcome.elapsed_s = round(time.monotonic() - started, 3)
        return outcome

    def _resolve_source(
        self,
        work: dict[str, Any],
        work_id: int,
        temp_dir: Path,
        timing: dict[str, float],
    ) -> Path:
        """解析源视频路径（本地直接使用；Bilibili 复用/写入 sources 持久缓存）。"""

        if work["source"] == "local":
            if not work.get("local_path"):
                raise PipelineError("本地作品缺少 local_path")
            source_path = self.config.local_work_path(str(work["local_path"]))
            if not source_path.is_file():
                raise PipelineError(f"本地视频不存在: {source_path}")
            self.catalog.begin_attempt(work_id, "encoding")
            return source_path
        if work["source"] == "bilibili":
            download_started = time.monotonic()
            source_path = self._download_bilibili_work(work, temp_dir)
            timing["download_s"] += time.monotonic() - download_started
            self.catalog.begin_attempt(work_id, "encoding")
            return source_path
        raise PipelineError(f"未知作品来源: {work['source']}")

    # ------------------------------------------------------------------ legacy 分支
    def _run_legacy(
        self,
        work_id: int,
        source_path: Path,
        info: Any,
        version: str,
        temp_dir: Path,
        timing: dict[str, float],
    ) -> int:
        """v0.3 行为：plan_clips 均匀抽 <=N 段 -> 编码 -> 平均 -> 登记。"""

        plans: tuple[ClipPlan, ...] = plan_clips(
            info.duration_s,
            clip_seconds=self.config.media.clip_seconds,
            clips_per_work=self.config.media.clips_per_work,
            min_duration_seconds=self.config.media.min_duration_seconds,
        )
        clip_dir = temp_dir / "clips"
        clip_dir.mkdir(parents=True, exist_ok=True)
        extract_started = time.monotonic()
        extracted = [
            extract_clip(source_path, plan, clip_dir, media=self.config.media, info=info)
            for plan in plans
        ]
        timing["extract_s"] += time.monotonic() - extract_started
        if self._sync_budget > 0:
            self._record_sync_checks(work_id, extracted)
            self._sync_budget -= 1

        batch = self.embedder.encode_pairs(
            [(clip.video_path, clip.audio_path) for clip in extracted]
        )
        timing["preprocess_s"] += float(getattr(batch, "preprocess_s", 0.0) or 0.0)
        timing["forward_s"] += float(getattr(batch, "forward_s", 0.0) or 0.0)
        self._validate_batch(batch, expected=len(extracted))

        save_started = time.monotonic()
        work_dir = self.config.paths.embeddings / version / f"{work_id:06d}"
        work_dir.mkdir(parents=True, exist_ok=True)
        saved = self._save_vectors(work_dir, batch)
        # 先登记片段计划再登记向量：这样崩溃在登记与 complete 之间时，
        # _heal_interrupted_work 能看到完整片段记录并安全自愈。
        self.catalog.replace_clips(
            work_id,
            version=version,
            clips=[(clip.index, clip.start_s, clip.duration_s) for clip in extracted],
        )
        self._register_vectors(work_id, version=version, saved=saved, n_clips=len(extracted))
        timing["save_s"] += time.monotonic() - save_started
        return len(extracted)

    # ------------------------------------------------------------------ full coverage 分支
    def _run_full_coverage(
        self,
        work_id: int,
        source_path: Path,
        info: Any,
        version: str,
        temp_dir: Path,
        timing: dict[str, float],
    ) -> int:
        """v0.4 全片覆盖：分片提取 -> 编码 -> 原子批文件 -> 拼接 -> 登记（支持恢复）。"""

        config = self.config
        duration_s = float(info.duration_s)
        if duration_s > config.media.long_video_seconds and not self.allow_long:
            raise PipelineError(
                f"超长视频 {duration_s:.1f}s 超过 media.long_video_seconds="
                f"{config.media.long_video_seconds:g}s；full_coverage 默认不处理，"
                "确认要全片处理请加 --allow-long"
            )
        plans = plan_full_coverage(
            duration_s,
            clip_seconds=config.media.clip_seconds,
            overlap_seconds=config.sampling.overlap_seconds,
            min_duration_seconds=config.media.min_duration_seconds,
        )
        n_clips = len(plans)
        batch_size = max(1, int(config.model.batch_size))
        work_dir = config.paths.embeddings / version / f"{work_id:06d}"
        partial_dir = work_dir / ".partial"
        partial_relpath = config.data_relpath(partial_dir)
        sampling_payload = {
            "mode": "full_coverage",
            "clip_seconds": float(config.media.clip_seconds),
            "overlap_seconds": float(config.sampling.overlap_seconds),
            "clip_height": config.media.clip_height,
            "duration_s": round(duration_s, 6),
            "event_version": config.events.version,
        }
        work_dir.mkdir(parents=True, exist_ok=True)
        done_batches = self._prepare_partial_plan(work_id, partial_dir, plans, sampling_payload)
        n_done = min(done_batches * batch_size, n_clips)
        self.catalog.upsert_sequence_state(
            work_id,
            version=version,
            duration_s=duration_s,
            n_clips=n_clips,
            n_clips_done=n_done,
            partial_relpath=partial_relpath,
            status="in_progress",
        )

        try:
            events_started = time.monotonic()
            self._scan_events_if_needed(work_id, work_dir, source_path, plans)
            timing["events_s"] += time.monotonic() - events_started

            completed_batches = 0
            for batch_index, chunk in enumerate(self._batched_plans(plans, batch_size)):
                batch_path = partial_dir / f"batch_{batch_index:04d}.npz"
                if batch_path.is_file():
                    completed_batches += 1
                    logger.info(
                        "work_id=%s full_coverage 复用已编码批次 %d/%d",
                        work_id,
                        batch_index + 1,
                        (n_clips + batch_size - 1) // batch_size,
                    )
                    continue
                clip_dir = temp_dir / f"batch_{batch_index:04d}"
                extract_started = time.monotonic()
                extracted = [
                    extract_clip(source_path, plan, clip_dir, media=config.media, info=info)
                    for plan in chunk
                ]
                timing["extract_s"] += time.monotonic() - extract_started
                try:
                    batch = self.embedder.encode_pairs(
                        [(clip.video_path, clip.audio_path) for clip in extracted]
                    )
                    timing["preprocess_s"] += float(
                        getattr(batch, "preprocess_s", 0.0) or 0.0
                    )
                    timing["forward_s"] += float(getattr(batch, "forward_s", 0.0) or 0.0)
                    self._validate_batch(batch, expected=len(chunk))
                    self._write_partial_batch(batch_path, batch)
                finally:
                    # 编码成功与否都删掉该批临时切片；失败批次不保留半成品编码
                    shutil.rmtree(clip_dir, ignore_errors=True)
                completed_batches += 1
                n_done = min(completed_batches * batch_size, n_clips)
                self.catalog.upsert_sequence_state(
                    work_id,
                    version=version,
                    duration_s=duration_s,
                    n_clips=n_clips,
                    n_clips_done=n_done,
                    partial_relpath=partial_relpath,
                    status="in_progress",
                )
                logger.info(
                    "work_id=%s full_coverage 批次 %d/%d 完成（%d/%d 片段）",
                    work_id,
                    batch_index + 1,
                    (n_clips + batch_size - 1) // batch_size,
                    n_done,
                    n_clips,
                )

            save_started = time.monotonic()
            saved = self._assemble_full_sequence(
                work_id=work_id,
                version=version,
                work_dir=work_dir,
                partial_dir=partial_dir,
                plans=plans,
                batch_size=batch_size,
                duration_s=duration_s,
                sampling_payload=sampling_payload,
            )
            self.catalog.replace_clips(
                work_id,
                version=version,
                clips=[
                    (
                        int(plan.index),
                        float(plan.start_s),
                        float(plan.duration_s),
                        float(plan.valid_duration_s),
                        float(plan.weight),
                    )
                    for plan in plans
                ],
            )
            self._register_vectors(work_id, version=version, saved=saved, n_clips=n_clips)
            self.catalog.upsert_sequence_state(
                work_id,
                version=version,
                duration_s=duration_s,
                n_clips=n_clips,
                n_clips_done=n_clips,
                partial_relpath=partial_relpath,
                status="assembled",
            )
            timing["save_s"] += time.monotonic() - save_started
        except Exception:
            try:
                done_now = min(
                    self._count_partial_batches(partial_dir) * batch_size, n_clips
                )
                self.catalog.upsert_sequence_state(
                    work_id,
                    version=version,
                    duration_s=duration_s,
                    n_clips=n_clips,
                    n_clips_done=done_now,
                    partial_relpath=partial_relpath,
                    status="failed",
                )
            except DatabaseError:  # pragma: no cover - 编目异常不掩盖主错误
                logger.exception("记录 work_id=%s 的失败序列状态时出错", work_id)
            raise

        shutil.rmtree(partial_dir, ignore_errors=True)
        return n_clips

    @staticmethod
    def _batched_plans(
        plans: Sequence[CoverageClip], batch_size: int
    ) -> list[list[CoverageClip]]:
        """把窗口计划按 batch_size 切成批次列表。"""

        return [
            list(plans[start : start + batch_size])
            for start in range(0, len(plans), batch_size)
        ]

    @staticmethod
    def _count_partial_batches(partial_dir: Path) -> int:
        """统计从 0 开始连续存在的 batch_*.npz 数量。"""

        count = 0
        while (partial_dir / f"batch_{count:04d}.npz").is_file():
            count += 1
        return count

    def _prepare_partial_plan(
        self,
        work_id: int,
        partial_dir: Path,
        plans: Sequence[CoverageClip],
        sampling_payload: Mapping[str, Any],
    ) -> int:
        """写入/校验 .partial/plan.json，返回已完成的批次数。

        若已有 plan.json 与当前窗口计划完全一致则复用（崩溃恢复）；不一致
        （源文件时长/采样参数变化）则丢弃整个 .partial 重新开始。
        """

        plan_path = partial_dir / "plan.json"
        expected: dict[str, Any] = {
            "work_id": work_id,
            "n_clips": len(plans),
            "sampling": dict(sampling_payload),
            "plans": [
                {
                    "index": int(plan.index),
                    "start_s": float(plan.start_s),
                    "duration_s": float(plan.duration_s),
                    "valid_duration_s": float(plan.valid_duration_s),
                    "weight": float(plan.weight),
                }
                for plan in plans
            ],
        }
        if plan_path.is_file():
            try:
                existing = json.loads(plan_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = None
            if existing == expected:
                return self._count_partial_batches(partial_dir)
            logger.warning(
                "work_id=%s 的 .partial/plan.json 与当前窗口计划不一致，丢弃 partial 重新编码",
                work_id,
            )
            shutil.rmtree(partial_dir, ignore_errors=True)
        partial_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write_json(plan_path, expected)
        return 0

    def _write_partial_batch(self, path: Path, batch: Any) -> None:
        """原子写入一个批次的向量 npz（先写同目录临时文件再 os.replace）。"""

        arrays: dict[str, np.ndarray] = {
            "joint": np.asarray(batch.joint, dtype=np.float32)
        }
        if getattr(batch, "audio", None) is not None:
            arrays["audio"] = np.asarray(batch.audio, dtype=np.float32)
        if getattr(batch, "video", None) is not None:
            arrays["video"] = np.asarray(batch.video, dtype=np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp")
        try:
            with tmp_path.open("wb") as handle:
                np.savez(handle, **arrays)
            os.replace(tmp_path, path)
        except OSError as exc:
            raise PipelineError(f"写入批次向量失败（{path}）：{exc}") from exc
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:  # pragma: no cover - 清理失败不改变主结果
                logger.warning("无法删除批次临时文件: %s", tmp_path)

    def _scan_events_if_needed(
        self,
        work_id: int,
        work_dir: Path,
        source_path: Path,
        plans: Sequence[CoverageClip],
    ) -> bool:
        """按需调用 nebula.events 扫描高频事件（延迟 import，失败让作品失败）。"""

        config = self.config
        if not config.events.enabled:
            return False
        scan_path = work_dir / "event_scan.npz"
        if scan_path.is_file():
            logger.info("work_id=%s 复用已有事件扫描文件: %s", work_id, scan_path)
            return False
        try:
            from .events import save_events_bundle, scan_events
        except ImportError as exc:
            raise PipelineError(
                "config.events.enabled=true 但 nebula.events 不可用："
                f"{exc}（事件扫描模块未交付或接口缺失）"
            ) from exc
        logger.info("work_id=%s 开始高频事件扫描（%d 个窗口）", work_id, len(plans))
        features = scan_events(
            source_path,
            plans,
            ffmpeg=config.media.ffmpeg,
            audio_rate_hz=int(config.events.audio_rate_hz),
            video_fps=int(config.events.video_fps),
            timeout_s=float(config.media.extract_timeout_seconds),
        )
        save_events_bundle(work_dir, features)
        if not scan_path.is_file():
            raise PipelineError(f"事件扫描未生成 event_scan.npz: {scan_path}")
        return True

    @staticmethod
    def _validate_sequence_times(
        plans: Sequence[CoverageClip], duration_s: float
    ) -> None:
        """校验拼接前的窗口时间：单调、无越界、无间隙、末尾锚定。"""

        if not plans:
            raise PipelineError("全片覆盖计划为空，无法拼接序列")
        starts = np.asarray([plan.start_s for plan in plans], dtype=np.float64)
        ends = starts + np.asarray([plan.duration_s for plan in plans], dtype=np.float64)
        if abs(float(starts[0])) > 1e-6:
            raise PipelineError(f"第一个窗口未从 0 开始: {starts[0]:.6f}")
        if abs(float(ends[-1]) - duration_s) > 1e-3:
            raise PipelineError(
                f"最后一个窗口结束于 {ends[-1]:.3f}s，未锚定视频结尾 {duration_s:.3f}s"
            )
        if float(starts.min()) < -1e-9 or float(ends.max()) > duration_s + 1e-3:
            raise PipelineError("窗口越界（起点为负或结束超过视频时长）")
        if np.any(np.diff(starts) < -1e-9):
            raise PipelineError("窗口起点不是单调非降")
        gaps = starts[1:] - ends[:-1]
        if np.any(gaps > 1e-6):
            raise PipelineError(
                f"窗口之间存在未覆盖间隙（最大 {float(gaps.max()):.6f}s）"
            )

    def _assemble_full_sequence(
        self,
        *,
        work_id: int,
        version: str,
        work_dir: Path,
        partial_dir: Path,
        plans: Sequence[CoverageClip],
        batch_size: int,
        duration_s: float,
        sampling_payload: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """拼接所有批次 -> 写 clip_*.npy / clip_times.npy / work_*.npy / manifest.json。"""

        from .embed import aggregate_work_vector, save_vector

        n_clips = len(plans)
        expected_batches = (n_clips + batch_size - 1) // batch_size
        joint_parts: list[np.ndarray] = []
        audio_parts: list[np.ndarray] = []
        video_parts: list[np.ndarray] = []
        has_audio: bool | None = None
        has_video: bool | None = None
        for batch_index in range(expected_batches):
            path = partial_dir / f"batch_{batch_index:04d}.npz"
            if not path.is_file():
                raise PipelineError(f"缺少批次向量文件，无法拼接: {path}")
            expected_rows = min(batch_size, n_clips - batch_index * batch_size)
            with np.load(path, allow_pickle=False) as payload:
                keys = set(payload.files)
                joint = np.asarray(payload["joint"], dtype=np.float32)
                if joint.ndim != 2 or joint.shape[0] != expected_rows or joint.shape[1] < 1:
                    raise EmbeddingError(
                        f"批次 {batch_index} 的 joint 形状异常: {joint.shape}（期望 {expected_rows} 行）"
                    )
                if not np.all(np.isfinite(joint)):
                    raise EmbeddingError(f"批次 {batch_index} 的 joint 含 NaN/Inf")
                joint_parts.append(joint)
                batch_has_audio = "audio" in keys
                batch_has_video = "video" in keys
                if has_audio is None:
                    has_audio, has_video = batch_has_audio, batch_has_video
                elif (batch_has_audio, batch_has_video) != (has_audio, has_video):
                    raise EmbeddingError(
                        f"批次 {batch_index} 的单模态向量可用性与其他批次不一致"
                    )
                if batch_has_audio:
                    audio = np.asarray(payload["audio"], dtype=np.float32)
                    if audio.shape != joint.shape:
                        raise EmbeddingError(f"批次 {batch_index} 的 audio 形状异常: {audio.shape}")
                    audio_parts.append(audio)
                if batch_has_video:
                    video = np.asarray(payload["video"], dtype=np.float32)
                    if video.shape != joint.shape:
                        raise EmbeddingError(f"批次 {batch_index} 的 video 形状异常: {video.shape}")
                    video_parts.append(video)

        clip_joint = np.concatenate(joint_parts, axis=0).astype(np.float32, copy=False)
        if clip_joint.shape[0] != n_clips:
            raise EmbeddingError(
                f"拼接后的片段数 {clip_joint.shape[0]} 与计划 {n_clips} 不一致"
            )
        clip_audio = (
            np.concatenate(audio_parts, axis=0).astype(np.float32, copy=False)
            if audio_parts
            else None
        )
        clip_video = (
            np.concatenate(video_parts, axis=0).astype(np.float32, copy=False)
            if video_parts
            else None
        )
        self._validate_sequence_times(plans, duration_s)

        saved: dict[str, dict[str, Any]] = {}

        def _save(kind: str, array: np.ndarray) -> None:
            path = work_dir / f"{kind}.npy"
            save_vector(path, array)
            saved[kind] = {
                "path": self.config.data_relpath(path),
                "dim": int(array.shape[-1]),
                "n_clips": int(array.shape[0]) if kind.startswith("clip_") else None,
            }

        _save("clip_joint", clip_joint)
        if clip_audio is not None:
            _save("clip_audio", clip_audio)
        if clip_video is not None:
            _save("clip_video", clip_video)
        _save("work_joint", aggregate_work_vector(clip_joint))
        if clip_audio is not None:
            _save("work_audio", aggregate_work_vector(clip_audio))
        if clip_video is not None:
            _save("work_video", aggregate_work_vector(clip_video))

        clip_times = np.asarray(
            [
                [
                    float(plan.start_s),
                    float(plan.duration_s),
                    float(plan.valid_duration_s),
                    float(plan.weight),
                ]
                for plan in plans
            ],
            dtype=np.float32,
        )
        save_vector(work_dir / "clip_times.npy", clip_times)

        manifest = {
            "work_id": work_id,
            "sequence_version": version,
            "sampling": dict(sampling_payload),
            "n_clips": n_clips,
            "duration_s": round(duration_s, 6),
            "clip_height": self.config.media.clip_height,
            "batch_size": batch_size,
            "event_version": self.config.events.version,
            "events_enabled": bool(self.config.events.enabled),
            "events_scanned": (work_dir / "event_scan.npz").is_file(),
            "embedder": type(self.embedder).__name__,
            "is_test_only": bool(getattr(self.embedder, "is_test_only", False)),
            "code_version": __version__,
            "created_at": now_utc_iso(),
        }
        self._atomic_write_json(work_dir / "manifest.json", manifest)
        logger.info(
            "work_id=%s 序列拼接完成：%d 个片段，权重和 %.6f，已写 manifest",
            work_id,
            n_clips,
            float(clip_times[:, 3].sum()),
        )
        return saved

    @staticmethod
    def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
        """以“同目录临时文件 + os.replace”原子写入 JSON。"""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_name(f".{target.name}.tmp")
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        try:
            tmp_path.write_text(text, encoding="utf-8")
            os.replace(tmp_path, target)
        except OSError as exc:
            raise PipelineError(f"写入 JSON 失败（{target}）：{exc}") from exc
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:  # pragma: no cover - 清理失败不改变主结果
                logger.warning("无法删除 JSON 临时文件: %s", tmp_path)

    def _download_bilibili_work(self, work: dict[str, Any], temp_dir: Path) -> Path:
        """获取 Bilibili 源视频：优先复用 sources 持久缓存，否则下载到 sources。

        v0.4.3.1：源文件不再放在 temp（会随作品完成被删除），而是持久缓存在
        data/sources/<bvid>/<bvid>_p<part>.*。缓存命中时跳过元数据请求与下载；
        未命中时先更新元数据，再延迟调用 t3 的 resolve_bilibili_source（下载到
        temp 暂存后原子移入 sources，失败清理暂存但不删已有缓存）。helper 尚未
        交付时回退到 pipeline 内的同语义实现。
        """

        if not work.get("bvid"):
            raise PipelineError("Bilibili 作品缺少 bvid")
        work_id = int(work["id"])
        self.catalog.begin_attempt(work_id, "downloading")
        bvid = str(work["bvid"])
        part = int(work["part"])

        cached = self._find_cached_bilibili_source(bvid, part)
        if cached is not None:
            logger.info(
                "work_id=%s 复用 sources 持久缓存源文件，跳过下载与元数据请求: %s",
                work_id,
                cached,
            )
            return cached

        from .download import BilibiliClient

        client = self._bilibili_client
        if client is None:
            client = BilibiliClient(self.config.bilibili)
            self._bilibili_client = client
        meta = client.fetch_meta(bvid, part=part)
        self.catalog.update_work_metadata(
            work_id,
            title=meta.title,
            author_uid=meta.author_uid,
            author_name=meta.author_name,
            duration_s=meta.duration_s,
            publish_date=meta.publish_date,
        )

        resolve = _load_resolve_bilibili_source()
        if resolve is not None:
            path = resolve(
                self.config, client, bvid, part, temp_dir, self.config.download
            )
        else:
            logger.warning(
                "nebula.download.resolve_bilibili_source 尚未就绪，"
                "使用 pipeline 内持久缓存回退（temp 暂存 -> 原子移入 sources）"
            )
            path = self._download_bilibili_source_fallback(bvid, part, temp_dir)
        return self._validate_bilibili_source_path(path, bvid=bvid, part=part)

    def _find_cached_bilibili_source(self, bvid: str, part: int) -> Path | None:
        """在 sources/<bvid>/ 下查找 bvid/part 匹配且可 probe 的持久缓存文件。"""

        bvid_dir = self.config.paths.sources / bvid
        if not bvid_dir.is_dir():
            return None
        for candidate in sorted(bvid_dir.glob(f"{bvid}_p{int(part)}.*")):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() in (
                ".part",
                ".tmp",
                ".ytdl",
                ".download",
                ".m4s",
            ):
                continue
            try:
                probe_media(
                    candidate,
                    ffprobe=self.config.media.ffprobe,
                    timeout_s=self.config.media.probe_timeout_seconds,
                )
            except MediaError:
                logger.debug("忽略不可 probe 的 sources 缓存文件: %s", candidate)
                continue
            return candidate
        return None

    def _validate_bilibili_source_path(
        self, path: Path, *, bvid: str, part: int
    ) -> Path:
        """校验 helper/回退返回的源文件存在且位于 sources 持久目录内。"""

        direct = Path(path)
        if not direct.is_file():
            raise PipelineError(f"下载后找不到文件（bvid={bvid}, part={part}）: {direct}")
        resolved = direct.resolve()
        sources_root = self.config.paths.sources.resolve()
        try:
            resolved.relative_to(sources_root)
        except ValueError as exc:
            raise PipelineError(
                f"Bilibili 源文件必须位于 sources 持久目录: {resolved}（sources={sources_root}）"
            ) from exc
        return resolved

    def _download_bilibili_source_fallback(
        self, bvid: str, part: int, temp_dir: Path
    ) -> Path:
        """helper 未就绪时的回退：temp 暂存下载后原子移入 sources/<bvid>/。"""

        from .download import download_bilibili

        staging = temp_dir / "download_staging"
        target_dir = self.config.paths.sources / bvid
        try:
            staging.mkdir(parents=True, exist_ok=True)
            path = download_bilibili(
                bvid,
                part=part,
                out_dir=staging,
                download=self.config.download,
            )
            if not path.is_file():
                raise PipelineError(f"下载后找不到文件: {path}")
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / path.name
            try:
                os.replace(path, target)
            except OSError:
                shutil.move(str(path), str(target))
            logger.info("Bilibili 源文件已写入持久缓存: %s", target)
            return target
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    # ------------------------------------------------------------------ 向量保存
    def _save_vectors(self, work_dir: Path, batch: Any) -> dict[str, dict[str, Any]]:
        from .embed import aggregate_work_vector, save_vector

        arrays: dict[str, np.ndarray] = {"work_joint": aggregate_work_vector(batch.joint)}
        if batch.audio is not None:
            arrays["work_audio"] = aggregate_work_vector(batch.audio)
        if batch.video is not None:
            arrays["work_video"] = aggregate_work_vector(batch.video)
        arrays["clip_joint"] = np.asarray(batch.joint, dtype=np.float32)
        if batch.audio is not None:
            arrays["clip_audio"] = np.asarray(batch.audio, dtype=np.float32)
        if batch.video is not None:
            arrays["clip_video"] = np.asarray(batch.video, dtype=np.float32)

        saved: dict[str, dict[str, Any]] = {}
        for kind, array in arrays.items():
            path = work_dir / f"{kind}.npy"
            save_vector(path, array)
            saved[kind] = {
                "path": self.config.data_relpath(path),
                "dim": int(array.shape[-1]),
                "n_clips": int(array.shape[0]) if kind.startswith("clip_") else None,
            }
        return saved

    def _register_vectors(
        self,
        work_id: int,
        *,
        version: str,
        saved: dict[str, dict[str, Any]],
        n_clips: int,
    ) -> None:
        for kind, meta in saved.items():
            expected_clips = n_clips if kind.startswith("clip_") else None
            self.catalog.register_embedding(
                work_id,
                version=version,
                kind=kind,
                path=meta["path"],
                dim=meta["dim"],
                n_clips=expected_clips,
            )

    def _validate_batch(self, batch: Any, *, expected: int) -> None:
        joint = getattr(batch, "joint", None)
        if joint is None:
            raise EmbeddingError("嵌入器没有返回联合向量")
        joint = np.asarray(joint)
        if joint.ndim != 2 or joint.shape[0] != expected:
            raise EmbeddingError(
                f"联合向量形状异常: 期望 ({expected}, dim)，实际 {joint.shape}"
            )
        if joint.shape[1] < 1:
            raise EmbeddingError("联合向量维度为 0")
        if not np.all(np.isfinite(joint)):
            raise EmbeddingError("联合向量包含 NaN/Inf")
        for name in ("audio", "video"):
            array = getattr(batch, name, None)
            if array is None:
                continue
            array = np.asarray(array)
            if array.ndim != 2 or array.shape[0] != expected:
                raise EmbeddingError(f"{name} 向量形状异常: {array.shape}")

    # ------------------------------------------------------------------ 音画同步抽查
    def _record_sync_checks(self, work_id: int, extracted: Sequence[Any]) -> None:
        for clip in extracted:
            try:
                report = clip_duration_report(
                    clip.video_path,
                    clip.audio_path,
                    ffprobe=self.config.media.ffprobe,
                )
            except MediaError as exc:  # pragma: no cover - 抽查失败不应阻塞处理
                report = {"ok": False, "error": str(exc)}
            report.update(
                {
                    "work_id": work_id,
                    "clip_index": clip.index,
                    "start_s": clip.start_s,
                    "planned_duration_s": clip.duration_s,
                }
            )
            self.sync_reports.append(report)

    def _write_sync_report(self) -> None:
        path = self.config.paths.logs / "sync_report.json"
        payload = {
            "generated_at": now_utc_iso(),
            "all_ok": all(bool(item.get("ok")) for item in self.sync_reports),
            "checks": self.sync_reports,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("音画同步抽查写入 %s", path)

    # ------------------------------------------------------------------ 消融实验
    def ablate(
        self,
        *,
        limit: int = 3,
        work_ids: Sequence[int] | None = None,
    ) -> list[dict[str, Any]]:
        """P0 排查实验：静音或遮蔽画面后，联合向量是否明显变化。

        对已完成且本地可访问的作品重新抽取片段，分别编码：
        原始音画 / 原画面+静音 / 纯黑画面+原音频，比较作品级联合向量余弦相似度。
        结果写入 logs/ablation_report.json；不写数据库、不覆盖正式向量。
        """

        from .embed import aggregate_work_vector

        if limit < 1:
            raise PipelineError("limit 必须 >= 1")
        self._ensure_version()
        if work_ids:
            works = []
            for work_id in work_ids:
                work = self.catalog.get_work(int(work_id))
                if work is None:
                    raise PipelineError(f"作品不存在: {work_id}")
                works.append(work)
        else:
            works = [
                w
                for w in self.catalog.list_works(status="completed")
                if w["source"] == "local" and w.get("local_path")
            ]
        results: list[dict[str, Any]] = []
        for work in works[:limit]:
            work_id = int(work["id"])
            temp_dir = self.config.paths.temp / f"ablate_{work_id:06d}"
            try:
                if work["source"] != "local":
                    raise PipelineError(
                        "消融实验目前只支持本地作品；Bilibili 作品请先归档原视频"
                    )
                source_path = self.config.local_work_path(str(work["local_path"]))
                if not source_path.is_file():
                    raise PipelineError(f"本地视频不存在: {source_path}")
                info = probe_media(
                    source_path,
                    ffprobe=self.config.media.ffprobe,
                    timeout_s=self.config.media.probe_timeout_seconds,
                )
                plans = plan_clips(
                    info.duration_s,
                    clip_seconds=self.config.media.clip_seconds,
                    clips_per_work=self.config.media.clips_per_work,
                    min_duration_seconds=self.config.media.min_duration_seconds,
                )
                clip_dir = temp_dir / "clips"
                clip_dir.mkdir(parents=True, exist_ok=True)
                extracted = [
                    extract_clip(source_path, plan, clip_dir, media=self.config.media)
                    for plan in plans
                ]
                silent_audio = [
                    mute_audio_clip(
                        clip.duration_s, clip_dir / f"silent_{clip.index:02d}.wav", media=self.config.media
                    )
                    for clip in extracted
                ]
                black_video = [
                    black_video_clip(
                        clip.duration_s, clip_dir / f"black_{clip.index:02d}.mp4", media=self.config.media
                    )
                    for clip in extracted
                ]
                normal_batch = self.embedder.encode_pairs(
                    [(clip.video_path, clip.audio_path) for clip in extracted]
                )
                mute_batch = self.embedder.encode_pairs(
                    [(clip.video_path, silent_audio[clip.index]) for clip in extracted]
                )
                black_batch = self.embedder.encode_pairs(
                    [(black_video[clip.index], clip.audio_path) for clip in extracted]
                )
                joint = aggregate_work_vector(normal_batch.joint)
                joint_mute = aggregate_work_vector(mute_batch.joint)
                joint_black = aggregate_work_vector(black_batch.joint)

                def cosine(a: np.ndarray, b: np.ndarray) -> float:
                    return float(np.clip(np.dot(a, b), -1.0, 1.0))

                results.append(
                    {
                        "work_id": work_id,
                        "title": work.get("title"),
                        "n_clips": len(extracted),
                        "duration_s": round(info.duration_s, 3),
                        "joint_vs_mute_cosine": round(cosine(joint, joint_mute), 4),
                        "joint_vs_black_cosine": round(cosine(joint, joint_black), 4),
                        "mute_vs_black_cosine": round(cosine(joint_mute, joint_black), 4),
                    }
                )
                logger.info("work_id=%s 消融完成: %s", work_id, results[-1])
            except Exception as exc:  # noqa: BLE001
                message = f"{type(exc).__name__}: {exc}"
                logger.exception("work_id=%s 消融失败: %s", work_id, message)
                results.append({"work_id": work_id, "error": message})
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        report_path = self.config.paths.logs / "ablation_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "generated_at": now_utc_iso(),
                    "embedding_version": self.embedding_version,
                    "model": self.config.model.checkpoint,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("消融报告写入 %s", report_path)
        return results

    # ------------------------------------------------------------------ 杂项
    def _temp_dir_count(self) -> int:
        if not self.config.paths.temp.is_dir():
            return 0
        return sum(1 for entry in self.config.paths.temp.iterdir() if entry.name != ".gitkeep")

