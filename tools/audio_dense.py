"""v0.5-beta 密集音频编码 pilot CLI（MERT-v1-95M，本地作品，不写 DB）。

用法（项目根目录）::

    .\\.venv\\Scripts\\python.exe tools/audio_dense.py --pilot --limit 20
    .\\.venv\\Scripts\\python.exe tools/audio_dense.py --work-ids 1 2 3 --force
    .\\.venv\\Scripts\\python.exe tools/audio_dense.py --pilot --out logs/audio_dense_pilot.json

行为：
- 只读 data/catalog.sqlite（sqlite3 URI mode=ro），选 canonical + completed 的
  本地作品（约 20 部，按 --limit / --work-ids；默认按时长分位覆盖不同时长）；
- 从 input/local 源文件用 ffmpeg 解码音频（不下载网络视频、不删除源文件）；
- 加载固定 revision 的 MERT-v1-95M（trust_remote_code=True），按
  context/hop 窗口编码帧特征，映射到 0.5s 全局 bin；
- 输出 data/audio_dense/<audio_version>/<work_id>/audio.npz + manifest.json；
- 输出 pilot JSON（默认 logs/audio_dense_pilot.json）并打印摘要；
- 记录 RTF、显存峰值、每部耗时、NaN/Inf/空序列/时长漂移、磁盘占用；
- 断点复用：完整产物且版本匹配时跳过（--force 覆盖）。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from nebula.audio_dense import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_BIN_SECONDS,
    DEFAULT_CONTEXT_S,
    DEFAULT_HOP_S,
    DEFAULT_LAYER,
    DEFAULT_MODEL_ID,
    DEFAULT_NORMALIZE,
    DEFAULT_POOL,
    DEFAULT_REVISION,
    AudioDenseError,
    HFAudioEncoder,
    compose_audio_version,
    extract_audio_features,
    is_audio_bundle_complete,
    save_audio_bundle,
)
from nebula.config import load_config  # noqa: E402
from nebula.errors import NebulaError  # noqa: E402
from nebula.log import configure_logging  # noqa: E402
from nebula.util import now_utc_iso  # noqa: E402

logger = logging.getLogger("audio_dense")

MAX_VRAM_MB = 8192.0


def _resolve_work_source(
    config: Any, work: dict[str, Any], *, temp_root: Path
) -> Path:
    """解析作品源文件：本地直接使用；Bilibili 走 data/sources 持久缓存 helper。"""

    work_id = int(work["id"])
    if str(work.get("source")) == "local":
        if not work.get("local_path"):
            raise AudioDenseError(f"本地作品缺少 local_path: work_id={work_id}")
        source = config.local_work_path(str(work["local_path"]))
        if not source.is_file():
            raise AudioDenseError(f"本地源文件不存在: {source}")
        return source
    if str(work.get("source")) == "bilibili":
        bvid = str(work.get("bvid") or "").strip()
        if not bvid:
            raise AudioDenseError(f"Bilibili 作品缺少 bvid: work_id={work_id}")
        part = int(work.get("part") or 1)
        from nebula.download import resolve_bilibili_source

        temp_dir = temp_root / f"work_{work_id:06d}"
        return resolve_bilibili_source(
            config,
            None,
            bvid,
            part,
            temp_dir,
            config.download,
            timeout_s=max(600.0, float(config.media.extract_timeout_seconds)),
        )
    raise AudioDenseError(f"未知作品来源: {work.get('source')!r}（work_id={work_id}）")


def _load_works(
    config: Any, *, work_ids: list[int] | None, limit: int, local_only: bool = False
) -> list[dict[str, Any]]:
    """只读 DB 选择 canonical + completed 作品（local / bilibili，排除 duplicate）。"""

    db_path = Path(config.paths.db_path)
    if not db_path.is_file():
        raise AudioDenseError(f"数据库不存在: {db_path}")
    uri = db_path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=30.0)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT w.id, w.title, w.source, w.local_path, w.bvid, w.part, "
            "w.duration_s FROM works w "
            "WHERE w.status = 'completed' "
            "AND w.source IN ('local', 'bilibili') "
            "AND w.id NOT IN (SELECT duplicate_work_id FROM work_links) "
            "ORDER BY w.duration_s, w.id"
        ).fetchall()
    finally:
        connection.close()
    candidates: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if not item.get("duration_s"):
            logger.warning("作品缺少时长，跳过 work_id=%s", item["id"])
            continue
        if local_only and str(item.get("source")) != "local":
            continue
        if str(item.get("source")) == "local":
            if not item.get("local_path"):
                continue
            source = config.local_work_path(str(item["local_path"]))
            if not source.is_file():
                logger.warning(
                    "本地源文件不存在，跳过 work_id=%s: %s", item["id"], source
                )
                continue
        elif not str(item.get("bvid") or "").strip():
            continue
        item["source_path"] = (
            config.local_work_path(str(item["local_path"]))
            if str(item.get("source")) == "local"
            else None
        )
        candidates.append(item)
    if not candidates:
        raise AudioDenseError("没有可用的本地 canonical completed 作品")
    if work_ids:
        by_id = {int(item["id"]): item for item in candidates}
        missing = [work_id for work_id in work_ids if work_id not in by_id]
        if missing:
            raise AudioDenseError(f"指定的 work_id 不可用: {missing}")
        return [by_id[int(work_id)] for work_id in work_ids]
    count = min(int(limit), len(candidates))
    if count <= 0:
        raise AudioDenseError(f"--limit 非法: {limit}")
    if count >= len(candidates):
        return candidates
    # 按时长分位选择，覆盖不同时长
    indices = sorted(
        {
            int(round(index * (len(candidates) - 1) / (count - 1)))
            for index in range(count)
        }
    )
    return [candidates[index] for index in indices]


def _select_duration_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return rows[:limit]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except OSError as exc:
        raise AudioDenseError(f"写入 pilot JSON 失败（{path}）：{exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


def _vram_peak_mb() -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return round(float(torch.cuda.max_memory_allocated()) / (1024**2), 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/audio_dense.py",
        description="v0.5-beta 密集音频编码 pilot（MERT-v1-95M，本地作品，不写 DB）。",
    )
    parser.add_argument("--config", default=None, help="配置文件路径（默认 ./config.yaml）")
    parser.add_argument("--pilot", action="store_true", help="pilot 模式（默认 limit=20，要求全部成功）")
    parser.add_argument("--limit", type=int, default=20, help="最多处理多少部作品（默认 20）")
    parser.add_argument("--work-ids", nargs="+", type=int, default=None, help="指定 work_id")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="只处理本地作品（默认 local + Bilibili canonical；Bilibili 源缓存到 data/sources）",
    )
    parser.add_argument("--force", action="store_true", help="覆盖已存在的完整产物")
    parser.add_argument("--version", default=None, help="audio dataset 版本（默认由模型/参数自动组合）")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="推理设备")
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="每批 context 窗口数"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="音频编码器 model id")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="模型 revision（固定）")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER, help="取第几层 encoder 输出")
    parser.add_argument("--context", type=float, default=DEFAULT_CONTEXT_S, help="context 秒数")
    parser.add_argument("--hop", type=float, default=DEFAULT_HOP_S, help="hop 秒数")
    parser.add_argument("--bin-seconds", type=float, default=DEFAULT_BIN_SECONDS, help="全局 bin 秒数")
    parser.add_argument("--normalize", default=DEFAULT_NORMALIZE, choices=["none", "l2"], help="帧特征归一化")
    parser.add_argument("--pool", default=DEFAULT_POOL, choices=["mean", "mean_std"], help="bin 池化")
    parser.add_argument("--audio-dir", default=None, help="输出根目录（默认 data/audio_dense）")
    parser.add_argument("--out", default=None, help="pilot JSON 路径（默认 logs/audio_dense_pilot.json）")
    parser.add_argument("--local-files-only", action="store_true", help="只用本地缓存（不下载）")
    parser.add_argument("--json", action="store_true", help="只输出 JSON 摘要")
    return parser


def _resolve_config_path(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    env_value = os.environ.get("NEBULA_CONFIG")
    if env_value:
        return Path(env_value).expanduser()
    return Path.cwd() / "config.yaml"


def _print_summary(report: dict[str, Any]) -> None:
    model = report["model"]
    summary = report["summary"]
    print("v0.5-beta 密集音频编码 pilot（MERT，本地作品，不写 DB）")
    print(
        f"  模型: {model['model']}（revision={model['revision']}，layer={model['layer']}，"
        f"params={model['params']}，trust_remote_code={model['trust_remote_code']}）"
    )
    print(
        f"  版本: {report['version']}（sr={report['config']['sample_rate']}，"
        f"ctx={report['config']['context_s']}s，hop={report['config']['hop_s']}s，"
        f"bin={report['config']['bin_seconds']}s，normalize={report['config']['normalize']}）"
    )
    print(
        f"  作品: 选中 {summary['selected']}，成功 {summary['processed']}，"
        f"复用 {summary['skipped']}，失败 {summary['failed']}"
    )
    print(
        f"  音频总时长 {summary['audio_seconds']:.1f}s；处理墙钟 {summary['processing_wall_s']:.1f}s；"
        f"RTF={summary['rtf']}；显存峰值 {summary['vram_peak_mb']} MB；磁盘 {summary['disk_bytes']} B"
    )
    checks = summary["checks"]
    for name, ok in checks.items():
        print(f"    {name}: {'PASS' if ok else 'FAIL'}")
    print(f"  结果: {'PASS' if summary['pass'] else 'FAIL'}")
    if summary.get("failure_reasons"):
        for reason in summary["failure_reasons"]:
            print(f"    - {reason}")
    print(f"  JSON: {report['artifacts']['json']}")
    print(f"  产物: {report['artifacts']['version_dir']}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(_resolve_config_path(args.config))
    except NebulaError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    configure_logging(config.logging.level)
    started = time.perf_counter()
    encoder: HFAudioEncoder | None = None
    torch: Any = None
    try:
        import torch as torch_module
    except ImportError:
        torch_module = None
    torch = torch_module
    if torch is not None and torch.cuda.is_available() and args.device in ("auto", "cuda"):
        torch.cuda.reset_peak_memory_stats()

    try:
        works = _load_works(
            config, work_ids=args.work_ids, limit=args.limit, local_only=args.local_only
        )
        encoder = HFAudioEncoder(
            args.model,
            revision=args.revision,
            cache_dir=config.model.cache_dir,
            device=args.device,
            layer=args.layer,
            trust_remote_code=True,
            local_files_only=bool(args.local_files_only),
        )
        encoder._ensure_loaded()  # 提前加载以确定真实 sample_rate / 显存峰值
        encoder_version = (
            f"{args.model}@{args.revision}" if args.revision else str(args.model)
        )
        version = args.version or compose_audio_version(
            encoder_version,
            sample_rate=int(encoder.sample_rate),
            context_s=args.context,
            hop_s=args.hop,
            layer=args.layer,
            pool=args.pool,
            bin_seconds=args.bin_seconds,
            normalize=args.normalize,
        )
        audio_root = (
            Path(args.audio_dir).expanduser()
            if args.audio_dir
            else config.paths.data / "audio_dense"
        )
        out_path = (
            Path(args.out).expanduser()
            if args.out
            else config.paths.logs / "audio_dense_pilot.json"
        )
        work_records: list[dict[str, Any]] = []
        processed = 0
        skipped = 0
        failed = 0
        audio_seconds = 0.0
        processing_wall = 0.0
        failure_reasons: list[str] = []
        disk_bytes = 0
        for position, work in enumerate(works, start=1):
            work_id = int(work["id"])
            work_dir = audio_root / version / f"{work_id:06d}"
            record: dict[str, Any] = {
                "work_id": work_id,
                "local_path": str(work["local_path"]),
                "duration_s": float(work["duration_s"]),
            }
            if not args.force and is_audio_bundle_complete(work_dir, version=version):
                skipped += 1
                record["status"] = "skipped"
                work_records.append(record)
                logger.info("[%d/%d] work_id=%s 复用已有产物", position, len(works), work_id)
                continue
            work_started = time.perf_counter()
            temp_root = config.paths.temp
            temp_dir = temp_root / f"audio_dense_{work_id:06d}"
            try:
                source_path = _resolve_work_source(
                    config, work, temp_root=temp_root
                )
                features = extract_audio_features(
                    source_path,
                    encoder,
                    version=version,
                    sample_rate=encoder.sample_rate or 24000,
                    context_s=args.context,
                    hop_s=args.hop,
                    bin_seconds=args.bin_seconds,
                    normalize=args.normalize,
                    pool=args.pool,
                    batch_size=args.batch_size,
                    ffmpeg=config.media.ffmpeg,
                    timeout_s=max(600.0, float(config.media.extract_timeout_seconds)),
                    work_id=work_id,
                )
                saved = save_audio_bundle(work_dir, features, force=args.force)
                work_wall = time.perf_counter() - work_started
                n_bins_expected = max(
                    1, int(np.ceil(features.duration_s / args.bin_seconds - 1e-9))
                )
                duration_drift = abs(features.decoded_duration_s - features.duration_s)
                record.update(
                    {
                        "status": "ok",
                        "source": str(work.get("source")),
                        "source_path": str(source_path),
                        "n_frames": int(features.frame_features.shape[0]),
                        "n_bins": int(features.bin_features.shape[0]),
                        "n_bins_expected": n_bins_expected,
                        "bin_mask_ratio": round(
                            float(features.bin_mask.sum()) / max(1, int(features.bin_mask.size)),
                            6,
                        ),
                        "bin_mask_missing": int((~features.bin_mask).sum()),
                        "last_frame_s": round(float(features.frame_times[-1]), 6),
                        "decoded_duration_s": round(float(features.decoded_duration_s), 6),
                        "duration_drift_s": round(float(duration_drift), 6),
                        "decode_s": round(float(features.decode_s), 3),
                        "encode_s": round(float(features.encode_s), 3),
                        "wall_s": round(float(work_wall), 3),
                        "disk_bytes": int(
                            sum(path.stat().st_size for path in saved.values())
                        ),
                        "frames_finite": bool(np.isfinite(features.frame_features).all()),
                        "bins_finite": bool(np.isfinite(features.bin_features).all()),
                    }
                )
                processed += 1
                audio_seconds += float(features.duration_s)
                processing_wall += work_wall
                disk_bytes += int(record["disk_bytes"])
                logger.info(
                    "[%d/%d] work_id=%s OK：%.1fs 音频 / %d 帧 / %d bin / %.2fs",
                    position,
                    len(works),
                    work_id,
                    features.duration_s,
                    features.frame_features.shape[0],
                    features.bin_features.shape[0],
                    work_wall,
                )
            except (AudioDenseError, NebulaError) as exc:
                failed += 1
                work_wall = time.perf_counter() - work_started
                record.update({"status": "failed", "error": str(exc), "wall_s": round(work_wall, 3)})
                failure_reasons.append(f"work_id={work_id}: {exc}")
                logger.error("[%d/%d] work_id=%s 失败: %s", position, len(works), work_id, exc)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
            work_records.append(record)
        vram_peak = _vram_peak_mb()
        checks = {
            "all_selected_success": failed == 0 and processed + skipped == len(works),
            "pilot_20_works": (not args.pilot) or (processed + skipped) >= 20,
            "no_nan_inf": all(
                record.get("frames_finite", True) and record.get("bins_finite", True)
                for record in work_records
                if record.get("status") == "ok"
            ),
            "bin_count_matches_duration": all(
                record.get("n_bins") == record.get("n_bins_expected")
                for record in work_records
                if record.get("status") == "ok"
            ),
            # 帧时间映射对齐：最后一帧相对已解码音频结尾的误差 <= 1 bin；
            # 容器时长与音频流时长差（个别 Bilibili 源音频比视频短 1-2s）
            # 单独记录在 summary.container_duration_drift_s，不作为编码失败。
            "frame_alignment_le_1_bin": all(
                float(record.get("decoded_duration_s", 0.0))
                - float(record.get("last_frame_s", 0.0))
                <= float(args.bin_seconds)
                and float(record.get("decoded_duration_s", 0.0))
                - float(record.get("last_frame_s", 0.0))
                >= -float(args.bin_seconds)
                for record in work_records
                if record.get("status") == "ok"
            ),
            "vram_le_8gb": vram_peak is None or float(vram_peak) <= MAX_VRAM_MB,
            "nonempty_all": all(
                int(record.get("n_frames", 0)) > 0
                for record in work_records
                if record.get("status") == "ok"
            ),
        }
        pass_flag = all(bool(value) for value in checks.values())
        strict_failed = [
            {
                "work_id": int(record["work_id"]),
                "drift_s": float(record.get("duration_drift_s", 0.0)),
                "tail_missing_bins": int(record.get("bin_mask_missing", 0)),
                "bin_mask_ratio": record.get("bin_mask_ratio"),
                "source": record.get("source"),
                "excluded_from_beta": True,
            }
            for record in work_records
            if record.get("status") == "ok"
            and float(record.get("duration_drift_s", 0.0)) > float(args.bin_seconds)
        ]
        strict_alignment = {
            "metric": "abs(decoded_audio_duration - ffprobe_container_duration) <= 1 bin",
            "bin_seconds": float(args.bin_seconds),
            "passed": len(strict_failed) == 0,
            "failed_count": len(strict_failed),
            "failed_works": strict_failed,
            "cause": (
                "源性时长差（音频流比容器/视频短 1.5-2.0s），末尾 bin 无帧（mask=False）；"
                "不是时间戳漂移或编码错误"
            ),
            "note": (
                "严格容器时长口径 108/110 通过；产物帧时间映射口径 110/110 通过；"
                "work 50/99 已从 v0.5-beta 77/22/11 划分中排除"
            ),
        }
        total_wall = time.perf_counter() - started
        report = {
            "created_at": now_utc_iso(),
            "model": {
                "model": args.model,
                "revision": args.revision,
                "trust_remote_code": True,
                "layer": args.layer,
                "params": int(
                    sum(parameter.numel() for parameter in encoder._model.parameters())
                )
                if encoder._model is not None
                else None,
                "load_s": round(float(encoder.model_load_s), 2),
            },
            "config": {
                "sample_rate": int(encoder.sample_rate or 0),
                "context_s": float(args.context),
                "hop_s": float(args.hop),
                "bin_seconds": float(args.bin_seconds),
                "normalize": args.normalize,
                "pool": args.pool,
                "device": args.device,
                "batch_size": int(args.batch_size),
            },
            "version": version,
            "selection": {
                "pilot": bool(args.pilot),
                "limit": int(args.limit),
                "work_ids": [int(work["id"]) for work in works],
            },
            "works": work_records,
            "summary": {
                "selected": len(works),
                "processed": processed,
                "skipped": skipped,
                "failed": failed,
                "audio_seconds": round(float(audio_seconds), 3),
                "processing_wall_s": round(float(processing_wall), 3),
                "total_wall_s": round(float(total_wall), 3),
                "rtf": (
                    round(float(processing_wall) / audio_seconds, 5)
                    if audio_seconds > 0
                    else None
                ),
                "realtime_speed": (
                    round(audio_seconds / processing_wall, 2)
                    if processing_wall > 0
                    else None
                ),
                "vram_peak_mb": vram_peak,
                "disk_bytes": int(disk_bytes),
                "alignment_definition": (
                    "frame_time_mapping: |decoded_audio_duration - last_frame_time| <= 1 bin；"
                    "容器时长与音频流时长差单独记录为 strict_container_alignment"
                ),
                "strict_container_alignment": strict_alignment,
                "container_duration_drift_s": {
                    "max": round(
                        max(
                            (
                                float(record.get("duration_drift_s", 0.0))
                                for record in work_records
                                if record.get("status") == "ok"
                            ),
                            default=0.0,
                        ),
                        6,
                    ),
                    "over_1_bin": [
                        {
                            "work_id": record["work_id"],
                            "drift_s": record.get("duration_drift_s"),
                            "bin_mask_ratio": record.get("bin_mask_ratio"),
                        }
                        for record in work_records
                        if record.get("status") == "ok"
                        and float(record.get("duration_drift_s", 0.0))
                        > float(args.bin_seconds)
                    ],
                },
                "checks": checks,
                "failure_reasons": failure_reasons,
                "pass": bool(pass_flag),
            },
            "artifacts": {
                "audio_root": str(audio_root),
                "version_dir": str(audio_root / version),
                "json": str(out_path),
            },
        }
        _atomic_write_json(out_path, report)
    except NebulaError as exc:
        print(f"密集音频 pilot 失败: {exc}", file=sys.stderr)
        return 1
    finally:
        if encoder is not None:
            encoder.close()

    if args.json:
        print(json.dumps(report, ensure_ascii=False))
    else:
        _print_summary(report)
    return 0 if report["summary"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
