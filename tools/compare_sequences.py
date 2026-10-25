"""比较两个 sequence version 的向量差异与耗时（v0.4 实验用，不重新编码）。

典型用途：
- 336p 预处理实验 vs 原生分辨率：同一批作品的全片向量余弦与耗时差异；
- legacy 5 片段平均 vs 全片平均：作品级向量的变化幅度（不同片段集合，仅作参考）。

用法（项目根目录，使用 .venv 中的 python）：

    python tools/compare_sequences.py --a peav-small-16f__f16_clip8s_x5_48k \
        --b peav-small-16f__f16_full8s_ov0_48k
    python tools/compare_sequences.py --a <480p版本> --b <h336版本> --csv logs/seq_h336.csv

输出 JSON：作品级余弦统计（joint/audio/video）、片段级对齐余弦（同片段数时）、
两个版本的 work_timings 分阶段均值。只读数据库与 npy，不加载 PE-AV。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nebula.config import Config, load_config
from nebula.db import Catalog
from nebula.embed import load_vector
from nebula.errors import NebulaError
from nebula.util import now_utc_iso

WORK_KINDS = ("work_joint", "work_audio", "work_video")


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _cutoff(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "p05": float(np.percentile(array, 5)),
        "std": float(array.std()),
    }


def _load_work_vectors(
    catalog: Catalog, config: Config, version: str, kind: str
) -> dict[int, np.ndarray]:
    rows = catalog.list_works_with_embedding(version, kind=kind)
    duplicates = catalog.get_duplicate_work_ids()
    result: dict[int, np.ndarray] = {}
    for row in rows:
        work_id = int(row["id"])
        if work_id in duplicates:
            continue
        result[work_id] = load_vector(config.data_path(str(row["embedding_path"])))
    return result


def _load_clip_matrices(
    catalog: Catalog, config: Config, version: str, kind: str
) -> dict[int, np.ndarray]:
    rows = catalog.list_embeddings(version, kind=kind)
    duplicates = catalog.get_duplicate_work_ids()
    result: dict[int, np.ndarray] = {}
    for row in rows:
        work_id = int(row["work_id"])
        if work_id in duplicates:
            continue
        result[work_id] = load_vector(config.data_path(str(row["path"])))
    return result


def _timing_summary(catalog: Catalog, version: str) -> dict[str, Any]:
    rows = catalog.list_work_timings(embedding_version=version)
    if not rows:
        return {"n": 0}
    fields = ("download_s", "probe_s", "extract_s", "preprocess_s", "forward_s", "save_s", "events_s", "total_s")
    summary: dict[str, Any] = {"n": len(rows)}
    for field in fields:
        values = [float(row.get(field) or 0.0) for row in rows]
        summary[field] = round(float(np.mean(values)), 4)
    summary["total_sum_s"] = round(sum(float(row.get("total_s") or 0.0) for row in rows), 3)
    return summary


def compare(
    config: Config,
    *,
    version_a: str,
    version_b: str,
    csv_path: Path | None = None,
) -> dict[str, Any]:
    catalog = Catalog(config.paths.db_path)
    catalog.initialize()
    try:
        for version in (version_a, version_b):
            if catalog.get_embedding_version(version) is None:
                raise NebulaError(f"sequence version 未登记: {version}")
        works = {int(row["id"]): row for row in catalog.list_works()}
        report: dict[str, Any] = {
            "generated_at": now_utc_iso(),
            "a": version_a,
            "b": version_b,
            "works_in_db": len(works),
            "work_cosine": {},
            "clip_cosine": {},
            "timings": {"a": _timing_summary(catalog, version_a), "b": _timing_summary(catalog, version_b)},
            "per_work": [],
        }
        common_ids: set[int] | None = None
        work_vectors: dict[str, dict[str, dict[int, np.ndarray]]] = {}
        for kind in WORK_KINDS:
            va = _load_work_vectors(catalog, config, version_a, kind)
            vb = _load_work_vectors(catalog, config, version_b, kind)
            work_vectors[kind] = {"a": va, "b": vb}
            ids = set(va) & set(vb)
            common_ids = ids if common_ids is None else common_ids & ids
        assert common_ids is not None
        common = sorted(common_ids)

        for kind in WORK_KINDS:
            va = work_vectors[kind]["a"]
            vb = work_vectors[kind]["b"]
            values = [_cos(va[wid], vb[wid]) for wid in common if wid in va and wid in vb]
            report["work_cosine"][kind] = _cutoff([v for v in values if np.isfinite(v)])

        clip_joint_a = _load_clip_matrices(catalog, config, version_a, "clip_joint")
        clip_joint_b = _load_clip_matrices(catalog, config, version_b, "clip_joint")
        aligned_values: list[float] = []
        aligned_works = 0
        for wid in common:
            ca = clip_joint_a.get(wid)
            cb = clip_joint_b.get(wid)
            if ca is None or cb is None or ca.shape != cb.shape:
                continue
            aligned_works += 1
            for row_a, row_b in zip(ca, cb, strict=True):
                value = _cos(row_a, row_b)
                if np.isfinite(value):
                    aligned_values.append(value)
        report["clip_cosine"] = {
            "aligned_works": aligned_works,
            "joint": _cutoff(aligned_values),
        }

        for wid in common:
            title = str(works.get(wid, {}).get("title") or works.get(wid, {}).get("work_key") or "")
            row: dict[str, Any] = {
                "work_id": wid,
                "title": title,
                "joint_cos": round(_cos(work_vectors["work_joint"]["a"][wid], work_vectors["work_joint"]["b"][wid]), 6),
                "audio_cos": round(_cos(work_vectors["work_audio"]["a"][wid], work_vectors["work_audio"]["b"][wid]), 6),
                "video_cos": round(_cos(work_vectors["work_video"]["a"][wid], work_vectors["work_video"]["b"][wid]), 6),
                "clips_a": int(clip_joint_a.get(wid, np.zeros((0,))).shape[0]),
                "clips_b": int(clip_joint_b.get(wid, np.zeros((0,))).shape[0]),
            }
            ca = clip_joint_a.get(wid)
            cb = clip_joint_b.get(wid)
            if ca is not None and cb is not None and ca.shape == cb.shape and ca.size:
                pair_values = [_cos(x, y) for x, y in zip(ca, cb, strict=True)]
                pair_values = [v for v in pair_values if np.isfinite(v)]
                row["clip_joint_cos_mean"] = round(float(np.mean(pair_values)), 6) if pair_values else None
            report["per_work"].append(row)

        if csv_path is not None:
            import csv

            csv_path.parent.mkdir(parents=True, exist_ok=True)
            columns = ["work_id", "title", "joint_cos", "audio_cos", "video_cos", "clips_a", "clips_b", "clip_joint_cos_mean"]
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(report["per_work"])
            report["csv_path"] = str(csv_path)
        return report
    finally:
        catalog.close()


def _print_summary(report: dict[str, Any]) -> None:
    print(f"A: {report['a']}")
    print(f"B: {report['b']}")
    print(f"共同作品: {report['work_cosine'].get('work_joint', {}).get('n', 0)}")
    for kind in WORK_KINDS:
        stats = report["work_cosine"].get(kind, {})
        if stats.get("n"):
            print(
                f"  {kind:10s} cos mean={stats['mean']:.6f} min={stats['min']:.6f} "
                f"p05={stats['p05']:.6f} max={stats['max']:.6f}"
            )
    clip_stats = report["clip_cosine"].get("joint", {})
    if clip_stats.get("n"):
        print(
            f"  片段对齐   n={clip_stats['n']} mean={clip_stats['mean']:.6f} "
            f"min={clip_stats['min']:.6f} p05={clip_stats['p05']:.6f}"
        )
    for label in ("a", "b"):
        timing = report["timings"][label]
        if timing.get("n"):
            print(
                f"  耗时 {label}: n={timing['n']} extract={timing['extract_s']:.3f}s "
                f"forward={timing['forward_s']:.3f}s events={timing.get('events_s', 0):.3f}s "
                f"total={timing['total_s']:.3f}s"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="比较两个 sequence version 的向量与耗时")
    parser.add_argument("--a", required=True, help="作为基准的 sequence version")
    parser.add_argument("--b", required=True, help="要对比的 sequence version")
    parser.add_argument("--config", default=None, help="配置文件路径（默认 ./config.yaml）")
    parser.add_argument("--out", default=None, help="JSON 输出路径（默认 logs/seq_compare_<a>_<b>.json）")
    parser.add_argument("--csv", default=None, help="逐作品 CSV 输出路径")
    args = parser.parse_args(argv)
    config = load_config(args.config or Path.cwd() / "config.yaml")
    out_path = (
        Path(args.out)
        if args.out
        else config.paths.logs
        / f"seq_compare_{args.a.split('__')[-1]}_{args.b.split('__')[-1]}.json"
    )
    try:
        report = compare(
            config,
            version_a=args.a,
            version_b=args.b,
            csv_path=Path(args.csv) if args.csv else None,
        )
    except NebulaError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_summary(report)
    print(f"JSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
