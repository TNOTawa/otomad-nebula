"""A3 片段稳定性报告：只用已保存向量，离线、只读、确定性。

对每部 canonical 作品，用已保存的片段级联合向量（clip_joint）做留一（leave-one-out）
重聚合，得到“去掉某一个片段后”的作品向量；在“其余作品仍使用正式作品向量（work_joint）”
的固定高维余弦空间里计算 Top-k 近邻，并与正式作品向量的 Top-k 比较：

- 重叠数 / 重叠率（overlap）：两次 Top-k 命中集合的交集大小；
- 最大名次变化（max_rank_shift）：正式 Top-k 邻居在留一排序里的名次漂移最大值；
- 平均余弦漂移（mean_cosine_drift）：正式 Top-k 邻居的相似度绝对值变化均值。

用途是回答“作品结论在多大程度上依赖某一个 8 秒片段”，用于风险评估与报告，
不是作品质量指标，也不参与 UMAP 坐标训练。

边界（与 docs/CONTRACTS_V2.md 第 6 节一致）：

- 只读 Catalog 与 data/embeddings 里已登记的 npy，不加载模型、不重新提取/编码；
- 不写数据库、不改任何向量文件；
- 只使用高维余弦相似度，不使用 UMAP 平面距离；
- duplicate 作品（身份关联中的重复实体）默认不参与评估，也不进入近邻空间。
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import Config
from .db import Catalog
from .embed import load_vector
from .errors import NebulaError
from .util import now_utc_iso

logger = logging.getLogger(__name__)

DEFAULT_K = 5
"""默认对比的 Top-k 近邻数。"""

DEFAULT_UNSTABLE_THRESHOLD = 0.8
"""平均 Top-k 重叠率低于该值视为“不稳定”作品。"""

_EPS = 1e-12

CSV_COLUMNS: tuple[str, ...] = (
    "work_id",
    "work_key",
    "title",
    "status",
    "n_clips",
    "n_loo",
    "k",
    "mean_overlap",
    "min_overlap",
    "max_rank_shift",
    "mean_cosine_drift",
    "unstable",
)
"""CSV 汇总表列（每部作品一行）。"""


# ──────────────────────────────────────────────────────────────────────────────
# 基础数学
# ──────────────────────────────────────────────────────────────────────────────


def _as_matrix(value: Any, *, what: str) -> np.ndarray:
    """把输入转成浮点二维矩阵并校验。"""

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise NebulaError(f"{what} 不是合法的数值矩阵: {exc}") from exc
    if array.ndim != 2:
        raise NebulaError(f"{what} 必须是二维矩阵，实际形状 {array.shape}")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise NebulaError(f"{what} 不能为空，实际形状 {array.shape}")
    if not np.all(np.isfinite(array)):
        raise NebulaError(f"{what} 含 NaN/Inf，无法计算稳定性")
    return array


def _unit_rows(matrix: np.ndarray, *, what: str) -> np.ndarray:
    """逐行 L2 归一化；零范数行直接报错（不静默跳过）。"""

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= _EPS):
        raise NebulaError(f"{what} 存在零范数行，无法归一化")
    return matrix / norms


def leave_one_out_vectors(clip_matrix: np.ndarray) -> np.ndarray:
    """计算留一重聚合向量：去掉第 i 个片段后其余片段“归一化-平均-归一化”。

    参数
    ----
    clip_matrix:
        形状 (n_clips, dim) 的片段级联合向量矩阵（内部会先逐行 L2 归一化，
        因此调用方是否已归一化都可以）。

    返回
    ----
    形状 (n_clips, dim) 的 float32 矩阵，第 i 行是去掉第 i 个片段后的作品向量，
    每行 L2 范数 ≈ 1。

    异常
    ----
    片段数 < 2、含非有限值、零范数片段、或去掉某片段后其余向量互相抵消（零向量）
    时抛 nebula.errors.NebulaError（带中文原因）。
    """

    matrix = _unit_rows(_as_matrix(clip_matrix, what="片段向量矩阵"), what="片段向量矩阵")
    n_clips = matrix.shape[0]
    if n_clips < 2:
        raise NebulaError(f"留一稳定性至少需要 2 个片段，实际 {n_clips} 个")

    total = matrix.sum(axis=0)
    result = np.empty_like(matrix)
    for index in range(n_clips):
        remaining = total - matrix[index]
        norm = float(np.linalg.norm(remaining))
        if norm <= _EPS:
            raise NebulaError(
                f"去掉第 {index} 个片段后其余片段向量互相抵消（零向量），无法计算留一稳定性"
            )
        result[index] = remaining / norm
    return result.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 近邻与留一对比
# ──────────────────────────────────────────────────────────────────────────────


def _ranked_neighbors(
    similarities: np.ndarray, work_ids: Sequence[int]
) -> list[tuple[int, float, int]]:
    """按 相似度降序、work_id 升序（并列时稳定）排序，返回 (work_id, 相似度, 名次)。"""

    order = sorted(
        range(len(work_ids)),
        key=lambda position: (-float(similarities[position]), int(work_ids[position])),
    )
    return [
        (int(work_ids[position]), float(similarities[position]), rank)
        for rank, position in enumerate(order, start=1)
    ]


def _evaluate_work(
    *,
    work_id: int,
    official_matrix: np.ndarray,
    work_ids: Sequence[int],
    position: int,
    clip_matrix: np.ndarray,
    k: int,
) -> dict[str, Any]:
    """对一部作品计算正式向量与全部留一向量之间的 Top-k 对比。"""

    n_works = len(work_ids)
    k_effective = min(int(k), n_works - 1)
    others_positions = [index for index in range(n_works) if index != position]
    others_matrix = official_matrix[others_positions]
    others_ids = [int(work_ids[index]) for index in others_positions]

    official_similarities = others_matrix @ official_matrix[position]
    official_ranked = _ranked_neighbors(official_similarities, others_ids)
    official_top = official_ranked[:k_effective]
    official_top_ids = {neighbor_id for neighbor_id, _similarity, _rank in official_top}

    loo_vectors = leave_one_out_vectors(clip_matrix)
    n_clips = int(clip_matrix.shape[0])

    loo_entries: list[dict[str, Any]] = []
    for removed_index in range(loo_vectors.shape[0]):
        loo_similarities = others_matrix @ loo_vectors[removed_index].astype(np.float64)
        loo_ranked = _ranked_neighbors(loo_similarities, others_ids)
        loo_rank = {neighbor_id: rank for neighbor_id, _s, rank in loo_ranked}
        loo_sim = {neighbor_id: similarity for neighbor_id, similarity, _r in loo_ranked}
        loo_top_ids = {
            neighbor_id for neighbor_id, _s, _r in loo_ranked[:k_effective]
        }
        overlap = len(official_top_ids & loo_top_ids)
        rank_shifts = [
            abs(loo_rank[neighbor_id] - rank)
            for neighbor_id, _similarity, rank in official_top
        ]
        cosine_drifts = [
            abs(loo_sim[neighbor_id] - similarity)
            for neighbor_id, similarity, _rank in official_top
        ]
        loo_entries.append(
            {
                "removed_clip_index": removed_index,
                "neighbors": [
                    {
                        "work_id": neighbor_id,
                        "similarity": round(similarity, 6),
                        "rank": rank,
                    }
                    for neighbor_id, similarity, rank in loo_ranked[:k_effective]
                ],
                "overlap": int(overlap),
                "overlap_ratio": round(overlap / k_effective, 6) if k_effective else 0.0,
                "max_rank_shift": int(max(rank_shifts)) if rank_shifts else 0,
                "mean_cosine_drift": (
                    round(float(np.mean(cosine_drifts)), 6) if cosine_drifts else 0.0
                ),
            }
        )

    overlap_ratios = [entry["overlap_ratio"] for entry in loo_entries]
    drifts = [entry["mean_cosine_drift"] for entry in loo_entries]
    mean_overlap = round(float(np.mean(overlap_ratios)), 6) if overlap_ratios else 0.0
    return {
        "work_id": int(work_id),
        "n_clips": n_clips,
        "n_loo": len(loo_entries),
        "official_neighbors": [
            {
                "work_id": neighbor_id,
                "similarity": round(similarity, 6),
                "rank": rank,
            }
            for neighbor_id, similarity, rank in official_top
        ],
        "mean_overlap": mean_overlap,
        "min_overlap": round(float(min(overlap_ratios)), 6) if overlap_ratios else 0.0,
        "max_rank_shift": max(
            (entry["max_rank_shift"] for entry in loo_entries), default=0
        ),
        "mean_cosine_drift": round(float(np.mean(drifts)), 6) if drifts else 0.0,
        "leave_one_out": loo_entries,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 报告生成
# ──────────────────────────────────────────────────────────────────────────────


def _load_work_vectors(
    cat: Catalog, config: Config, rows: Sequence[Any], version: str
) -> tuple[list[int], np.ndarray]:
    """读取 canonical 作品的正式 work_joint 向量，返回排序后的 (ids, 已归一化矩阵)。"""

    work_ids: list[int] = []
    vectors: list[np.ndarray] = []
    dim: int | None = None
    for row in rows:
        work_id = int(row["id"])
        path = config.data_path(str(row["embedding_path"]))
        try:
            vector = np.asarray(load_vector(path), dtype=np.float64)
        except NebulaError as exc:
            raise NebulaError(f"读取作品 {work_id} 的 work_joint 向量失败: {exc}") from exc
        if vector.ndim != 1:
            raise NebulaError(
                f"作品 {work_id} 的 work_joint 维度异常（期望一维，实际 {vector.shape}）"
            )
        if not np.all(np.isfinite(vector)):
            raise NebulaError(f"作品 {work_id} 的 work_joint 含 NaN/Inf")
        if dim is None:
            dim = int(vector.shape[0])
        elif int(vector.shape[0]) != dim:
            raise NebulaError(
                f"向量版本 {version} 内作品维度不一致: {work_id} 为 {vector.shape[0]}，期望 {dim}"
            )
        work_ids.append(work_id)
        vectors.append(vector)
    if not vectors:
        raise NebulaError(f"向量版本 {version} 没有任何可用的 canonical work_joint 向量")
    matrix = _unit_rows(np.stack(vectors), what="作品向量")
    return work_ids, matrix


def stability_report(
    cat: Catalog,
    config: Config,
    *,
    version: str | None = None,
    k: int = DEFAULT_K,
    work_ids: Sequence[int] | None = None,
    unstable_threshold: float = DEFAULT_UNSTABLE_THRESHOLD,
) -> dict[str, Any]:
    """生成片段稳定性报告（JSON 可序列化的 dict）。

    参数
    ----
    cat:
        已初始化的 nebula.db.Catalog（只读使用）。
    config:
        配置对象；默认向量版本取 config.embedding_version。
    version:
        向量版本；省略时使用配置文件派生的当前版本。
    k:
        对比的 Top-k 近邻数；作品不足时自动截断到 n_canonical - 1。
    work_ids:
        只评估指定作品；省略时评估全部 canonical 作品。未知/重复/非 canonical 的 id
        会进入 skipped 而不是报错。
    unstable_threshold:
        判定“不稳定”的重叠率阈值（默认 0.8）；仅影响汇总字段。

    返回
    ----
    包含 embedding_version / k / 汇总指标 / works 明细 / skipped 的 dict。
    """

    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise NebulaError(f"k 必须是 >= 1 的整数，实际 {k!r}")
    threshold = float(unstable_threshold)
    if not 0.0 < threshold <= 1.0:
        raise NebulaError(f"不稳定阈值必须在 (0, 1] 内，实际 {unstable_threshold!r}")

    resolved_version = str(version) if version else config.embedding_version
    rows = cat.list_works_with_embedding(resolved_version, kind="work_joint")
    if not rows:
        raise NebulaError(
            f"向量版本 {resolved_version} 没有已登记的 work_joint 向量，无法生成稳定性报告"
        )
    duplicate_ids = cat.get_duplicate_work_ids()
    canonical_rows = [row for row in rows if int(row["id"]) not in duplicate_ids]
    if len(canonical_rows) < 2:
        raise NebulaError(
            f"可用于稳定性分析的 canonical 作品不足（{len(canonical_rows)} 部），至少需要 2 部"
        )
    logger.info(
        "片段稳定性分析: version=%s canonical=%d（已排除 %d 个 duplicate）",
        resolved_version,
        len(canonical_rows),
        len(rows) - len(canonical_rows),
    )

    work_ids_all, official_matrix = _load_work_vectors(
        cat, config, canonical_rows, resolved_version
    )
    position_of = {work_id: index for index, work_id in enumerate(work_ids_all)}
    rows_by_id = {int(row["id"]): row for row in canonical_rows}

    skipped: list[dict[str, Any]] = []
    if work_ids is None:
        evaluate_ids = list(work_ids_all)
    else:
        evaluate_ids = []
        seen: set[int] = set()
        for raw_work_id in work_ids:
            work_id = int(raw_work_id)
            if work_id in seen:
                skipped.append({"work_id": work_id, "reason": "重复指定，已忽略"})
                continue
            seen.add(work_id)
            if work_id not in position_of:
                skipped.append(
                    {
                        "work_id": work_id,
                        "reason": "不是 canonical 作品，或缺少当前向量版本的 work_joint",
                    }
                )
                continue
            evaluate_ids.append(work_id)

    works: list[dict[str, Any]] = []
    for index, work_id in enumerate(evaluate_ids, start=1):
        row = rows_by_id[work_id]
        clip_row = cat.get_embedding(work_id, version=resolved_version, kind="clip_joint")
        if clip_row is None:
            skipped.append(
                {
                    "work_id": work_id,
                    "work_key": row.get("work_key"),
                    "reason": "缺少 clip_joint 片段向量，无法留一",
                }
            )
            continue
        try:
            clip_matrix = np.asarray(
                load_vector(config.data_path(str(clip_row["path"]))), dtype=np.float64
            )
        except NebulaError as exc:
            skipped.append(
                {
                    "work_id": work_id,
                    "work_key": row.get("work_key"),
                    "reason": f"读取 clip_joint 失败: {exc}",
                }
            )
            continue
        if clip_matrix.ndim != 2 or clip_matrix.shape[0] < 2:
            count = clip_matrix.shape[0] if clip_matrix.ndim == 2 else 0
            skipped.append(
                {
                    "work_id": work_id,
                    "work_key": row.get("work_key"),
                    "reason": f"片段数不足（{count} 个），无法留一",
                }
            )
            continue
        if clip_matrix.shape[1] != official_matrix.shape[1]:
            skipped.append(
                {
                    "work_id": work_id,
                    "work_key": row.get("work_key"),
                    "reason": (
                        f"片段向量维度 {clip_matrix.shape[1]} 与作品向量维度 "
                        f"{official_matrix.shape[1]} 不一致"
                    ),
                }
            )
            continue
        try:
            evaluated = _evaluate_work(
                work_id=work_id,
                official_matrix=official_matrix,
                work_ids=work_ids_all,
                position=position_of[work_id],
                clip_matrix=clip_matrix,
                k=k,
            )
        except NebulaError as exc:
            skipped.append(
                {
                    "work_id": work_id,
                    "work_key": row.get("work_key"),
                    "reason": f"留一计算失败: {exc}",
                }
            )
            continue
        evaluated.update(
            {
                "work_key": row.get("work_key"),
                "title": row.get("title"),
                "status": row.get("status"),
            }
        )
        works.append(evaluated)
        if index % 10 == 0 or index == len(evaluate_ids):
            logger.info("片段稳定性分析进度: %d/%d", index, len(evaluate_ids))

    works.sort(key=lambda item: int(item["work_id"]))
    if works:
        mean_overlap = round(float(np.mean([item["mean_overlap"] for item in works])), 6)
        mean_cosine_drift = round(
            float(np.mean([item["mean_cosine_drift"] for item in works])), 6
        )
        unstable_count = sum(1 for item in works if item["mean_overlap"] < threshold)
    else:
        mean_overlap = 0.0
        mean_cosine_drift = 0.0
        unstable_count = 0
    for item in works:
        item["unstable"] = bool(item["mean_overlap"] < threshold)

    report: dict[str, Any] = {
        "generated_at": now_utc_iso(),
        "embedding_version": resolved_version,
        "k": int(k),
        "effective_k": min(int(k), len(work_ids_all) - 1),
        "unstable_threshold": threshold,
        "n_work_joint": len(rows),
        "n_canonical_works": len(canonical_rows),
        "n_excluded_duplicates": len(rows) - len(canonical_rows),
        "n_evaluated": len(works),
        "n_skipped": len(skipped),
        "mean_overlap": mean_overlap,
        "mean_cosine_drift": mean_cosine_drift,
        "unstable_count": int(unstable_count),
        "works": works,
        "skipped": skipped,
        "caveats": [
            "只用已保存向量计算，不加载模型、不重新编码",
            "不参与 UMAP 坐标训练，也不作为作品质量指标",
            "使用高维余弦相似度，不使用 UMAP 平面距离",
            "duplicate 作品默认不参与评估，也不进入近邻空间",
        ],
    }
    return report


# ──────────────────────────────────────────────────────────────────────────────
# 报告落盘
# ──────────────────────────────────────────────────────────────────────────────


def _atomic_write_text(path: Path, text: str) -> None:
    """以“同目录临时文件 + os.replace”原子写入文本。"""

    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise NebulaError(f"无法创建报告目录 {target.parent}: {exc}") from exc
    tmp_path: Path | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(tmp_path, target)
    except OSError as exc:
        raise NebulaError(f"写入报告失败（{target}）: {exc}") from exc
    finally:
        if tmp_path is not None:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:  # pragma: no cover - 清理失败不改变主结果
                logger.warning("无法删除报告临时文件: %s", tmp_path)


def _report_csv(report: dict[str, Any]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    for item in report.get("works", []):
        writer.writerow(
            {
                "work_id": item.get("work_id"),
                "work_key": item.get("work_key"),
                "title": item.get("title"),
                "status": item.get("status"),
                "n_clips": item.get("n_clips"),
                "n_loo": item.get("n_loo"),
                "k": report.get("effective_k", report.get("k")),
                "mean_overlap": item.get("mean_overlap"),
                "min_overlap": item.get("min_overlap"),
                "max_rank_shift": item.get("max_rank_shift"),
                "mean_cosine_drift": item.get("mean_cosine_drift"),
                "unstable": "true" if item.get("unstable") else "false",
            }
        )
    return buffer.getvalue()


def write_report(
    report: dict[str, Any], *, json_path: str | Path, csv_path: str | Path
) -> None:
    """把稳定性报告写成 JSON（完整明细）与 CSV（每部作品一行汇总）。

    两个文件都先写同目录临时文件再原子替换；失败抛带中文原因的
    nebula.errors.NebulaError，不会留下半个文件。
    """

    if not isinstance(report, dict):
        raise NebulaError("稳定性报告必须是 dict")
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    json_text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    csv_text = _report_csv(report)
    _atomic_write_text(json_target, json_text)
    _atomic_write_text(csv_target, csv_text)
