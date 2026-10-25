r"""v0.4.2 表征构建 CLI：Full Mean / Temporal Stats。

只读取序列层已保存的片段向量（data/embeddings/<sequence_version>/），
不下载、不解码、不加载 PE-AV、不访问网络；重复构建不覆盖旧序列版本。

用法（项目根目录）::

    .\.venv\Scripts\python.exe tools/build_representation.py list
    .\.venv\Scripts\python.exe tools/build_representation.py full_mean --sequence auto
    .\.venv\Scripts\python.exe tools/build_representation.py temporal_stats --sequence full --stages 4
    .\.venv\Scripts\python.exe tools/build_representation.py temporal_stats --sequence <version> --limit 10
    .\.venv\Scripts\python.exe tools/build_representation.py full_mean --sequence <version> --force

行为：
- 只处理 canonical（排除 work_links duplicate）且有完整 clip 序列的作品；
- sequence_state 存在且不是 assembled 的作品跳过；缺事件特征的作品在 events 变体下跳过；
- 构建前登记/复用 representation，状态置 building；成功后置 ready 并记录 dim；
- 无 --force 时增量构建（已登记向量跳过）；--force 先清空该表征全部 vectors 再全量重建；
- 结束打印 n_built / n_existing / n_skipped / n_errors 与耗时。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nebula.config import Config, load_config  # noqa: E402
from nebula.db import Catalog  # noqa: E402
from nebula.errors import NebulaError  # noqa: E402
from nebula.log import configure_logging  # noqa: E402
from nebula.representations import (  # noqa: E402
    FULL_MEAN_V1,
    TEMPORAL_STATS_V1,
    load_clip_sequence,
    normalize_representation_name,
    representation_label,
    save_representation_vectors,
    sequence_work_dir,
)
from nebula.temporal import (  # noqa: E402
    EVENT_FEATURE_DIM,
    assemble_full_mean,
    assemble_temporal_stats,
    full_mean_params,
    temporal_stats_params,
    vector_dim,
)

LOGGER = logging.getLogger("build_representation")

SEQUENCE_AUTO = "auto"
SEQUENCE_ALIASES = {"auto": None, "legacy": "legacy", "full": "full"}

#: CLI 子命令 -> 表征名/DB kind
BUILD_COMMANDS: dict[str, tuple[str, str]] = {
    "full_mean": (FULL_MEAN_V1, "full_mean"),
    "temporal_stats": (TEMPORAL_STATS_V1, "temporal_stats"),
}

DEFAULT_CONFIG = "config.yaml"


# --------------------------------------------------------------------------- 参数解析
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/build_representation.py",
        description="v0.4.2 表征构建：从已保存的 clip 序列重建 Full Mean / Temporal Stats。",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径（默认当前目录 config.yaml）",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--sequence",
            default=SEQUENCE_AUTO,
            help="序列版本：auto（当前 config.embedding_version）/ legacy / full / 显式版本号",
        )
        sub.add_argument("--limit", type=int, default=None, help="最多构建多少部作品")
        sub.add_argument(
            "--force",
            action="store_true",
            help="重建：清空该表征已登记的 vectors 后全量重写（可与旧参数不同）",
        )

    full = subparsers.add_parser("full_mean", help="Full Mean（全片片段平均）")
    add_common(full)

    temporal = subparsers.add_parser("temporal_stats", help="Temporal Stats（固定时间结构）")
    add_common(temporal)
    temporal.add_argument(
        "--stages",
        type=int,
        default=None,
        help="相对时间阶段数（默认取 config.temporal.stages）",
    )

    subparsers.add_parser("list", help="列出已登记的表征及其向量/地图数量")
    return parser


# --------------------------------------------------------------------------- 序列与候选
def resolve_sequence_version(catalog: Catalog, config: Config, value: str | None) -> str:
    """把 auto/legacy/full/显式版本号解析为已登记的 sequence_version。"""

    raw = (value or SEQUENCE_AUTO).strip()
    if raw in ("", SEQUENCE_AUTO):
        version = config.embedding_version
    elif raw == "legacy":
        version = config.legacy_embedding_version
    elif raw == "full":
        version = config.full_sequence_version
    else:
        version = raw
    if catalog.get_embedding_version(version) is None:
        available = _registered_sequence_versions(catalog)
        raise NebulaError(
            f"序列版本未登记: {version}（已登记: {', '.join(available) or '无'}）。"
            "请先用 pipeline.py 编码该 sequence version，或显式传 --sequence <version>"
        )
    return version


def _registered_sequence_versions(catalog: Catalog) -> list[str]:
    rows = catalog.conn.execute(
        "SELECT version FROM embedding_versions ORDER BY version"
    ).fetchall()
    return [str(row["version"]) for row in rows]


def _sequence_metadata(
    catalog: Catalog, sequence_version: str, work_id: int
) -> dict[str, Any] | None:
    """不读 npy 的序列完整性预检；不完整返回 None。"""

    joint = catalog.get_embedding(work_id, version=sequence_version, kind="clip_joint")
    if joint is None:
        return None
    clips = catalog.get_clips(work_id, version=sequence_version)
    n_clips = joint.get("n_clips")
    n_clips = len(clips) if n_clips is None else int(n_clips)
    if n_clips < 1 or len(clips) != n_clips:
        return None
    clip_dim = int(joint["dim"])
    for kind in ("clip_audio", "clip_video"):
        row = catalog.get_embedding(work_id, version=sequence_version, kind=kind)
        if row is None:
            continue
        if int(row["dim"]) != clip_dim:
            return None
        row_clips = row.get("n_clips")
        if row_clips is not None and int(row_clips) != n_clips:
            return None
    return {"n_clips": n_clips, "clip_dim": clip_dim}


def discover_candidates(
    catalog: Catalog, sequence_version: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """返回 (可构建作品元数据, 跳过明细, 排除的 duplicate 数量)。

    可构建 = canonical + clip_joint 已登记 + 序列记录完整 + sequence_state 为
    assembled（没有 state 的旧序列视为完整，兼容 legacy 数据）。
    """

    duplicates = catalog.get_duplicate_work_ids()
    states = {
        int(state["work_id"]): state
        for state in catalog.list_sequence_states(version=sequence_version)
    }
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    excluded_duplicates = 0
    for row in catalog.list_embeddings(sequence_version, kind="clip_joint"):
        work_id = int(row["work_id"])
        if work_id in duplicates:
            excluded_duplicates += 1
            continue
        state = states.get(work_id)
        if state is not None and str(state["status"]) != "assembled":
            skipped.append(
                {"work_id": work_id, "reason": f"sequence_state={state['status']}"}
            )
            continue
        metadata = _sequence_metadata(catalog, sequence_version, work_id)
        if metadata is None:
            skipped.append({"work_id": work_id, "reason": "序列记录不完整"})
            continue
        candidates.append({"work_id": work_id, **metadata})
    return candidates, skipped, excluded_duplicates


# --------------------------------------------------------------------------- 事件变体
def _load_event_features(
    config: Config, sequence_version: str, work_id: int, n_clips: int
) -> np.ndarray | None:
    path = sequence_work_dir(config, sequence_version, work_id) / "event_features.npy"
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        LOGGER.warning("work_id=%s 事件特征读取失败，按缺失处理: %s", work_id, exc)
        return None
    if array.ndim != 2 or array.shape[0] != n_clips or array.shape[1] != EVENT_FEATURE_DIM:
        LOGGER.warning(
            "work_id=%s 事件特征形状异常 %s，按缺失处理", work_id, array.shape
        )
        return None
    if not np.all(np.isfinite(array)):
        LOGGER.warning("work_id=%s 事件特征含 NaN/Inf，按缺失处理", work_id)
        return None
    return np.asarray(array, dtype=np.float32)


def _event_manifest_version(
    config: Config, sequence_version: str, work_id: int
) -> str | None:
    path = sequence_work_dir(config, sequence_version, work_id) / "event_manifest.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    version = payload.get("version")
    return str(version) if version else None


def determine_event_variant(
    catalog: Catalog,
    config: Config,
    sequence_version: str,
    candidates: Sequence[dict[str, Any]],
) -> tuple[bool, str | None, set[int]]:
    """按设计契约判定事件变体：至少一部有有效事件 -> 含事件变体。"""

    versions: set[str] = set()
    with_events: set[int] = set()
    for item in candidates:
        work_id = int(item["work_id"])
        features = _load_event_features(
            config, sequence_version, work_id, int(item["n_clips"])
        )
        if features is None:
            continue
        with_events.add(work_id)
        version = _event_manifest_version(config, sequence_version, work_id)
        if version:
            versions.add(version)
    if not with_events:
        return False, None, set()
    if len(versions) > 1:
        raise NebulaError(
            "同一序列版本下存在多个事件版本，禁止混用: "
            + ", ".join(sorted(versions))
        )
    event_version = next(iter(versions)) if versions else config.events.version
    return True, event_version, with_events


# --------------------------------------------------------------------------- 构建
def build_representation(
    catalog: Catalog,
    config: Config,
    *,
    name: str,
    sequence: str | None = None,
    stages: int | None = None,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """从已保存 clip 序列构建一个表征并登记，返回构建摘要。"""

    normalized = normalize_representation_name(name)
    if normalized not in (FULL_MEAN_V1, TEMPORAL_STATS_V1):
        raise NebulaError(
            f"不支持的表征: {name}（可用: full_mean / temporal_stats）"
        )
    kind = "full_mean" if normalized == FULL_MEAN_V1 else "temporal_stats"
    stage_count = (
        int(stages) if stages is not None else int(config.temporal.stages)
    )
    if normalized == TEMPORAL_STATS_V1:
        temporal_stats_params(stages=stage_count)  # 参数校验（非法 stages 尽早失败）
    if limit is not None and limit < 1:
        raise NebulaError("--limit 必须 >= 1")
    if force and limit is not None:
        raise NebulaError("--force 会清空该表征的全部 vectors，不能与 --limit 同时使用")

    started = time.monotonic()
    sequence_version = resolve_sequence_version(catalog, config, sequence)
    candidates, skipped, excluded_duplicates = discover_candidates(
        catalog, sequence_version
    )
    if not candidates:
        raise NebulaError(
            f"序列版本 {sequence_version} 没有可构建的 canonical 完整序列；"
            f"跳过 {len(skipped)} 部"
        )

    has_events, event_version, with_events = determine_event_variant(
        catalog, config, sequence_version, candidates
    )
    if has_events:
        missing_events = [
            item for item in candidates if int(item["work_id"]) not in with_events
        ]
        for item in missing_events:
            skipped.append(
                {
                    "work_id": int(item["work_id"]),
                    "reason": "缺少有效事件特征（events 变体要求）",
                }
            )
        candidates = [
            item for item in candidates if int(item["work_id"]) in with_events
        ]
    if limit is not None:
        candidates = candidates[:limit]
    if not candidates:
        raise NebulaError("没有可构建的作品（事件变体下所有作品都缺事件特征）")

    clip_dims = {int(item["clip_dim"]) for item in candidates}
    if len(clip_dims) != 1:
        raise NebulaError(f"候选作品的 clip 维度不一致: {sorted(clip_dims)}")
    clip_dim = int(next(iter(clip_dims)))

    if normalized == FULL_MEAN_V1:
        params = full_mean_params(clip_dim=clip_dim, sequence_version=sequence_version)
        expected_dim = clip_dim
    else:
        params = temporal_stats_params(
            stages=stage_count,
            has_events=has_events,
            event_version=event_version,
            clip_dim=clip_dim,
            sequence_version=sequence_version,
        )
        expected_dim = vector_dim(
            "joint", clip_dim=clip_dim, stages=stage_count, has_events=has_events
        )

    note = (
        f"v0.4.2 {representation_label(normalized)}；sequence={sequence_version}；"
        f"stages={stage_count}；has_events={has_events}"
    )
    prefix = (
        f"representations/{normalized}/{sequence_version}"
    )
    existing = catalog.get_representation(
        name=normalized, sequence_version=sequence_version
    )
    if existing is None:
        representation_id = catalog.register_representation(
            normalized,
            sequence_version=sequence_version,
            kind=kind,
            params=params,
            path_prefix=prefix,
            note=note,
        )
        existing_work_ids: set[int] = set()
    else:
        representation_id = int(existing["id"])
        if force:
            catalog.update_representation_params(representation_id, params, note=note)
            cleared = catalog.clear_representation_vectors(representation_id)
            LOGGER.info("--force 清空表征 %s 的 %d 条 vectors", normalized, cleared)
            existing_work_ids = set()
        else:
            try:
                representation_id = catalog.register_representation(
                    normalized,
                    sequence_version=sequence_version,
                    kind=kind,
                    params=params,
                    path_prefix=prefix,
                    note=note,
                )
            except NebulaError as exc:
                raise NebulaError(
                    f"{exc}；如确认按新参数重建，请加 --force"
                ) from exc
            existing_work_ids = {
                int(row["work_id"])
                for row in catalog.list_representation_vectors(
                    representation_id, kind="joint"
                )
            }
    rep = catalog.get_representation(representation_id)
    assert rep is not None
    catalog.set_representation_status(representation_id, "building", note=note)

    built = 0
    existing_skipped = 0
    errors: list[dict[str, Any]] = []
    total = len(candidates)
    for position, item in enumerate(candidates, start=1):
        work_id = int(item["work_id"])
        if work_id in existing_work_ids:
            existing_skipped += 1
            print(f"  [{position}/{total}] work_id={work_id} 已存在，跳过")
            continue
        work_started = time.monotonic()
        try:
            seq = load_clip_sequence(catalog, config, sequence_version, work_id)
            if normalized == FULL_MEAN_V1:
                arrays = assemble_full_mean(seq)
            else:
                arrays = assemble_temporal_stats(seq, stages=stage_count)
            saved = save_representation_vectors(
                catalog, config, rep, work_id, arrays
            )
            elapsed = time.monotonic() - work_started
            dims = {kind_name: int(info["dim"]) for kind_name, info in saved.items()}
            event_work = work_id in with_events and normalized == TEMPORAL_STATS_V1
            print(
                f"  [{position}/{total}] work_id={work_id} dims "
                f"{' '.join(f'{k}:{v}' for k, v in dims.items())}"
                f"{'（含事件）' if event_work else ''} {elapsed:.2f}s"
            )
            LOGGER.info(
                "work_id=%s 表征 %s 构建完成 dims=%s 耗时 %.2fs",
                work_id,
                normalized,
                dims,
                elapsed,
            )
            built += 1
        except NebulaError as exc:
            elapsed = time.monotonic() - work_started
            message = f"{type(exc).__name__}: {exc}"
            errors.append({"work_id": work_id, "error": message})
            print(f"  [{position}/{total}] work_id={work_id} 失败: {message} ({elapsed:.2f}s)")
            LOGGER.error("work_id=%s 表征构建失败: %s", work_id, message)

    summary: dict[str, Any] = {
        "name": normalized,
        "label": representation_label(normalized),
        "kind": kind,
        "representation_id": representation_id,
        "sequence_version": sequence_version,
        "stages": stage_count,
        "has_events": has_events,
        "event_version": event_version if has_events else None,
        "clip_dim": clip_dim,
        "dim": expected_dim,
        "n_candidates": len(candidates),
        "n_built": built,
        "n_existing": existing_skipped,
        "n_skipped": len(skipped),
        "n_errors": len(errors),
        "n_excluded_duplicates": excluded_duplicates,
        "elapsed_s": round(time.monotonic() - started, 3),
        "skipped": skipped,
        "errors": errors,
        "status": "failed" if errors else "ready",
    }

    if errors:
        catalog.set_representation_status(
            representation_id,
            "failed",
            note=f"{len(errors)} 部作品构建失败；修复后可用 --force 重建",
        )
        return summary

    catalog.set_representation_status(
        representation_id, "ready", dim=expected_dim, note=note
    )
    return summary


# --------------------------------------------------------------------------- list
def list_representation_rows(catalog: Catalog) -> list[dict[str, Any]]:
    """列出台账（含各模态向量数与 ready 地图数），供 CLI/测试使用。"""

    rows: list[dict[str, Any]] = []
    for rep in catalog.list_representations():
        representation_id = int(rep["id"])
        counts = {"joint": 0, "audio": 0, "video": 0}
        for vector in catalog.list_representation_vectors(representation_id):
            kind = str(vector["kind"])
            counts[kind] = counts.get(kind, 0) + 1
        maps = catalog.list_maps(representation_id=representation_id)
        rows.append(
            {
                **rep,
                "label": representation_label(str(rep["name"])),
                "n_joint": counts["joint"],
                "n_audio": counts["audio"],
                "n_video": counts["video"],
                "n_maps_ready": sum(1 for item in maps if item.get("status") == "ready"),
            }
        )
    return rows


def print_representation_rows(rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        print("没有任何已登记的表征。")
        return
    print(
        "ID  名称                         序列版本                              "
        "类型            状态     dim   joint/audio/video  maps"
    )
    for row in rows:
        print(
            f"{int(row['id']):<3} {str(row['name']):<28} {str(row['sequence_version']):<36} "
            f"{str(row['kind']):<15} {str(row['status']):<7} "
            f"{int(row['dim']):<5} "
            f"{int(row['n_joint'])}/{int(row['n_audio'])}/{int(row['n_video']):<9} "
            f"{int(row['n_maps_ready'])}"
        )


def print_build_summary(summary: dict[str, Any]) -> None:
    print("")
    print(f"表征构建{'完成' if summary['status'] == 'ready' else '未完成'}: {summary['name']}")
    print(
        f"  id={summary['representation_id']}  sequence={summary['sequence_version']}  "
        f"status={summary['status']}  dim={summary['dim']}"
    )
    print(
        f"  stages={summary['stages']}  has_events={summary['has_events']}  "
        f"event_version={summary['event_version']}"
    )
    print(
        f"  候选={summary['n_candidates']}  新建={summary['n_built']}  "
        f"已存在={summary['n_existing']}  数据跳过={summary['n_skipped']}  "
        f"失败={summary['n_errors']}  duplicate排除={summary['n_excluded_duplicates']}"
    )
    print(f"  耗时={summary['elapsed_s']:.2f}s")
    if summary["skipped"]:
        print(f"  跳过明细（前 10 条，共 {len(summary['skipped'])}）:")
        for item in summary["skipped"][:10]:
            print(f"    work_id={item['work_id']}: {item['reason']}")
    if summary["errors"]:
        print(f"  失败明细（共 {len(summary['errors'])}）:")
        for item in summary["errors"]:
            print(f"    work_id={item['work_id']}: {item['error']}")


# --------------------------------------------------------------------------- 入口
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config) if args.config else Path.cwd() / DEFAULT_CONFIG
    try:
        config = load_config(config_path)
    except NebulaError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    configure_logging(config.logging.level)

    catalog = Catalog(config.paths.db_path)
    try:
        catalog.initialize()
        if args.command == "list":
            print_representation_rows(list_representation_rows(catalog))
            return 0
        name = FULL_MEAN_V1 if args.command == "full_mean" else TEMPORAL_STATS_V1
        summary = build_representation(
            catalog,
            config,
            name=name,
            sequence=args.sequence,
            stages=getattr(args, "stages", None),
            limit=args.limit,
            force=args.force,
        )
        print_build_summary(summary)
        if summary["n_errors"]:
            print(
                f"有 {summary['n_errors']} 部作品构建失败，表征状态置 failed；"
                "修复后可用 --force 重建。",
                file=sys.stderr,
            )
            return 1
        return 0
    except NebulaError as exc:
        print(f"表征构建失败: {exc}", file=sys.stderr)
        return 1
    finally:
        catalog.close()


if __name__ == "__main__":
    sys.exit(main())
