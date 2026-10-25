"""流水线：下载 -> 切片 -> 推理 -> 保存 -> 登记 -> 清理。

处理一部作品的顺序（任何一步失败都会记入 failed 并保留原因）：

1. 解析来源：本地文件直接使用（永不删除）；Bilibili 作品先取元数据再下载到 temp。
2. ffprobe 读取时长与音视频流信息，按 nebula.sampling.plan_clips 生成均匀片段。
3. ffmpeg 在 temp 中截取音画同步片段（mp4 + wav）。
4. 调用 Embedder 得到片段级联合/音频/视频向量。
5. 逐行归一化 -> 求平均 -> 再归一化，得到作品级向量。
6. 写入 data/embeddings/<向量版本>/<work_id>/ 并登记到 SQLite（先写文件再登记）。
7. 登记成功后删除该作品在 temp 中的全部临时文件；失败时保留 temp 便于排查。

断点续跑：已有当前向量版本 work_joint 向量的作品直接跳过；换模型/采样参数会生成新版本，
老作品会被 reopen 并补算新向量，旧版本文件保留。
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

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
from .sampling import ClipPlan, plan_clips
from .util import now_utc_iso, slugify

logger = logging.getLogger(__name__)

CLIP_SUFFIX = "__f{frames}_clip{clip_seconds}s_x{clips}_{sr}k"


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
    ) -> None:
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
        """由嵌入器版本 + 采样参数拼出的向量版本；不同版本绝不覆盖。"""

        if self._version is None:
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
        params = {
            "embedder": type(embedder).__name__,
            "is_test_only": is_test,
            "aggregation": "l2-normalize-mean-l2-normalize",
            "preprocess": "ffmpeg 8s clip h264/yuv420p + wav 48k/2ch/pcm_s16le",
            "ablation": None,
        }
        self.catalog.register_embedding_version(
            version,
            model_checkpoint=(
                self.config.model.checkpoint if not is_test else f"test-only:{embedder.version}"
            ),
            model_revision=None if is_test else self.config.model.revision,
            frames=self.config.model.frames,
            clip_seconds=self.config.media.clip_seconds,
            clips_per_work=self.config.media.clips_per_work,
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
            "save_s": 0.0,
        }
        try:
            if str(work["status"]) in ("completed", "downloading", "encoding"):
                self.catalog.reopen(
                    work_id, reason=f"为向量版本 {version} 重新处理（旧向量保留）"
                )
            if work["source"] == "local":
                if not work.get("local_path"):
                    raise PipelineError("本地作品缺少 local_path")
                source_path = self.config.local_work_path(str(work["local_path"]))
                if not source_path.is_file():
                    raise PipelineError(f"本地视频不存在: {source_path}")
                self.catalog.begin_attempt(work_id, "encoding")
            elif work["source"] == "bilibili":
                download_started = time.monotonic()
                source_path = self._download_bilibili_work(work, temp_dir)
                timing["download_s"] = time.monotonic() - download_started
                self.catalog.begin_attempt(work_id, "encoding")
            else:
                raise PipelineError(f"未知作品来源: {work['source']}")

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
                extract_clip(
                    source_path,
                    plan,
                    clip_dir,
                    media=self.config.media,
                    info=info,
                )
                for plan in plans
            ]
            timing["extract_s"] = time.monotonic() - extract_started
            if self._sync_budget > 0:
                self._record_sync_checks(work_id, extracted)
                self._sync_budget -= 1

            batch = self.embedder.encode_pairs(
                [(clip.video_path, clip.audio_path) for clip in extracted]
            )
            timing["preprocess_s"] = float(getattr(batch, "preprocess_s", 0.0) or 0.0)
            timing["forward_s"] = float(getattr(batch, "forward_s", 0.0) or 0.0)
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
            self.catalog.complete(work_id)
            timing["save_s"] = time.monotonic() - save_started
            outcome = WorkOutcome(
                work_id=work_id,
                status="completed",
                n_clips=len(extracted),
                duration_s=round(info.duration_s, 3),
            )
            # 只有向量成功写入并登记后才清理临时文件
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info("work_id=%s 完成：%d 个片段，%.1fs", work_id, len(extracted), info.duration_s)
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

    def _download_bilibili_work(self, work: dict[str, Any], temp_dir: Path) -> Path:
        if not work.get("bvid"):
            raise PipelineError("Bilibili 作品缺少 bvid")
        self.catalog.begin_attempt(int(work["id"]), "downloading")
        from .download import BilibiliClient, download_bilibili

        client = self._bilibili_client
        if client is None:
            client = BilibiliClient(self.config.bilibili)
            self._bilibili_client = client
        bvid = str(work["bvid"])
        part = int(work["part"])
        meta = client.fetch_meta(bvid, part=part)
        self.catalog.update_work_metadata(
            int(work["id"]),
            title=meta.title,
            author_uid=meta.author_uid,
            author_name=meta.author_name,
            duration_s=meta.duration_s,
            publish_date=meta.publish_date,
        )
        download_dir = temp_dir / "download"
        download_dir.mkdir(parents=True, exist_ok=True)
        path = download_bilibili(
            bvid,
            part=part,
            out_dir=download_dir,
            download=self.config.download,
        )
        if not path.is_file():
            raise PipelineError(f"下载后找不到文件: {path}")
        return path

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

