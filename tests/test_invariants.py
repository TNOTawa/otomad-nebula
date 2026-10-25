"""独立架构 / 语义不变量测试（QA 视角）。

这些测试约束跨模块的契约，而不是复刻某个实现方的单元测试：

- 模块导入边界：可视化数据层不依赖 UI 库；向量提取不依赖 pandas / 元数据编目；
- 元数据不进入向量的存储与接口（作品标题、时长、作者、榜单等只用于展示）；
- 向量版本号随模型 / 采样参数变化，且不含元数据；
- 处理状态机的可达性与终态约束；
- work_key 去重规则（本地相对路径、bilibili bvid+part）；
- 采样规则不产生负数 / 越界片段，且段数受上限约束；
- .gitignore 覆盖运行时产物与虚拟环境；
- 全部 CLI 入口的 --help 可运行。

对尚未交付的模块使用 pytest.skip，而不是让并行开发期间整套测试变红；
交付验收时 qa.py 的入口检查会强制要求这些模块存在。
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import random
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nebula import CLIP_VECTOR_KINDS, EMBEDDING_KINDS, WORK_VECTOR_KINDS
from nebula.config import Config
from nebula.db import (
    SCHEMA_VERSION,
    STATUSES,
    TRANSITIONS,
    Catalog,
    work_key_for_bilibili,
    work_key_for_local,
)
from nebula.embed import save_vector
from nebula.errors import DatabaseError, NebulaError, SamplingError
from nebula.representations import (
    load_representation_matrix,
    resolve_representation,
    save_representation_vectors,
)
from nebula.sampling import plan_clips
from nebula.util import slugify

REPO_ROOT = Path(__file__).resolve().parents[1]

#: 元数据字段：只允许存在于 works / lists 等展示层，不允许进入向量。
METADATA_FIELDS = (
    "title",
    "author_uid",
    "author_name",
    "duration_s",
    "publish_date",
    "rank",
    "votes",
    "vote_year",
)


# ──────────────────────────────────────────────────────────────────────────────
# 静态导入边界
# ──────────────────────────────────────────────────────────────────────────────


def _module_file(module_name: str) -> Path:
    path = REPO_ROOT / "nebula" / f"{module_name}.py"
    if not path.is_file():
        pytest.skip(f"模块 nebula/{module_name}.py 尚未交付")
    return path


def _imported_modules(path: Path) -> set[str]:
    """解析源文件，返回 import 到的模块全名（相对导入还原为 nebula.*）。"""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.level:
                modules.add("nebula." + node.module)
            else:
                modules.add(node.module)
    return modules


def _assert_no_import(path: Path, forbidden: tuple[str, ...]) -> None:
    imported = _imported_modules(path)
    violations = sorted(
        name
        for name in imported
        if any(name == item or name.startswith(item + ".") for item in forbidden)
    )
    assert not violations, f"{path.name} 违反模块边界，导入了: {violations}"


def test_appdata_does_not_import_ui_libraries() -> None:
    """appdata 必须是纯数据层（契约：不导入 streamlit / plotly）。"""

    path = _module_file("appdata")
    _assert_no_import(path, ("streamlit", "plotly"))


def test_embed_does_not_import_metadata_or_pandas() -> None:
    """向量提取只依赖音视频与数值库，不依赖 pandas / SQLite 元数据 / UI。"""

    path = _module_file("embed")
    _assert_no_import(
        path,
        ("pandas", "streamlit", "plotly", "nebula.db", "nebula.appdata", "nebula.mapplot"),
    )


def test_mapplot_does_not_import_ui_libraries() -> None:
    """UMAP 星云生成是纯计算，不应依赖 Streamlit / Plotly。"""

    path = _module_file("mapplot")
    _assert_no_import(path, ("streamlit", "plotly"))


def test_similarity_uses_high_dim_not_umap() -> None:
    """A1 近邻索引必须使用高维余弦；禁止依赖 UMAP 平面距离或模型推理。"""

    path = _module_file("similarity")
    _assert_no_import(path, ("umap", "streamlit", "plotly", "torch", "transformers"))


def test_identity_logic_is_offline() -> None:
    """B1 身份关联逻辑不加载模型/界面库；网络客户端通过参数注入。"""

    path = _module_file("identity")
    _assert_no_import(path, ("torch", "transformers", "streamlit", "plotly", "umap"))


def test_stability_report_is_offline_and_plan_distance_free() -> None:
    """A3 稳定性只读已保存向量：不加载模型、不导入 UMAP/界面库。"""

    path = _module_file("stability")
    _assert_no_import(
        path, ("torch", "transformers", "streamlit", "plotly", "umap", "pandas")
    )
    tool = REPO_ROOT / "tools" / "clip_stability.py"
    if not tool.is_file():
        pytest.skip("tools/clip_stability.py 尚未交付")
    _assert_no_import(tool, ("torch", "transformers", "streamlit", "plotly", "umap"))


def test_avsync_import_boundaries() -> None:
    """A6 错位实验模块：不导入 UI 库，导入模块本身不得加载 torch/transformers。"""

    path = _module_file("avsync")
    _assert_no_import(path, ("streamlit", "plotly"))
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, nebula.avsync; "
                "print(int('torch' in sys.modules), int('transformers' in sys.modules), "
                "int('streamlit' in sys.modules), int('plotly' in sys.modules))"
            ),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr[-800:]
    assert probe.stdout.strip() == "0 0 0 0", f"导入 nebula.avsync 意外加载了模型/UI 库: {probe.stdout}"


def test_similarity_explanation_fields_are_display_only() -> None:
    """A1 近邻解释字段（相对分位/互为）有默认值、不改变既有构造与排序语义。"""

    module = importlib.import_module("nebula.similarity")
    neighbor_cls = module.Neighbor
    plain = neighbor_cls(work_id=1, similarity=0.5, rank=1)
    assert plain.pair_percentile == 0.0
    assert plain.is_mutual is False
    detailed = neighbor_cls(
        work_id=2,
        similarity=0.1,
        rank=2,
        pair_percentile=42.5,
        is_mutual=True,
    )
    assert detailed.pair_percentile == 42.5
    assert detailed.is_mutual is True
    assert (detailed.work_id, detailed.similarity, detailed.rank) == (2, 0.1, 2)


def test_segment_explorer_is_pure_data_layer() -> None:
    """片段级探索接口位于纯数据层（appdata/similarity），不接收 UI 对象。"""

    appdata_path = _module_file("appdata")
    for function_name in ("hub_rows", "clip_heatmap_data", "clip_pair_rows"):
        params = _function_param_names(appdata_path, function_name)
        assert params is not None, f"appdata 缺少纯数据函数 {function_name}"
        assert not set(params) & {"fig", "figure", "st", "plotly"}, (
            f"{function_name} 不应接收 UI 对象参数: {params}"
        )
    similarity_path = _module_file("similarity")
    for method_name in (
        "pair_percentile",
        "hubness",
        "hub_top",
        "clip_similarity",
        "best_clip_pairs",
        "explain_pair",
    ):
        assert _function_param_names(similarity_path, method_name) is not None, (
            f"similarity 缺少 {method_name}"
        )
    # appdata 仍是纯数据层（不 import streamlit/plotly）
    _assert_no_import(appdata_path, ("streamlit", "plotly"))


# ──────────────────────────────────────────────────────────────────────────────
# 元数据不进入向量
# ──────────────────────────────────────────────────────────────────────────────


def test_embedding_registration_api_has_no_metadata_fields() -> None:
    """向量登记接口只接受版本 / 类型 / 路径 / 维度，不接受作品元数据。"""

    params = set(inspect.signature(Catalog.register_embedding).parameters)
    assert params == {"self", "work_id", "version", "kind", "path", "dim", "n_clips"}
    version_params = set(inspect.signature(Catalog.register_embedding_version).parameters)
    assert not version_params & set(METADATA_FIELDS), (
        f"向量版本登记接口不应接受元数据字段: {sorted(version_params & set(METADATA_FIELDS))}"
    )


def test_embeddings_table_has_no_metadata_columns(catalog: Catalog) -> None:
    """SQLite 层：embeddings 表只登记向量路径与维度，元数据只留在 works 表。"""

    columns = {
        str(row[1]) for row in catalog.conn.execute("PRAGMA table_info(embeddings)").fetchall()
    }
    leaked = columns & set(METADATA_FIELDS)
    assert not leaked, f"embeddings 表出现了元数据列: {sorted(leaked)}"
    assert {"path", "dim", "version", "kind"} <= columns
    works_columns = {
        str(row[1]) for row in catalog.conn.execute("PRAGMA table_info(works)").fetchall()
    }
    assert {"title", "duration_s", "author_uid"} <= works_columns


def _function_param_names(path: Path, function_name: str) -> list[str] | None:
    """用 AST 读取函数参数名（避免为了一个签名导入 umap/torch 等重依赖）。"""

    tree = ast.parse(path.read_text(encoding="utf-8"))

    def _match(node: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node
        if isinstance(node, (ast.ClassDef, ast.Module)):
            for child in node.body:
                found = _match(child)
                if found is not None:
                    return found
        return None

    node = _match(tree)
    if node is None:
        return None
    all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
    names = [arg.arg for arg in all_args]
    if node.args.vararg is not None:
        names.append(node.args.vararg.arg)
    if node.args.kwarg is not None:
        names.append(node.args.kwarg.arg)
    return names


def test_embedder_interface_takes_paths_only() -> None:
    """Embedder.encode_pairs 只接收 (视频, 音频) 路径对，不接收作品元数据。"""

    path = _module_file("embed")
    params = _function_param_names(path, "encode_pairs")
    if params is None:
        pytest.skip("nebula.embed 尚无 encode_pairs")
    assert params == ["self", "pairs"], f"encode_pairs 参数异常: {params}"


def test_map_loader_interface_has_no_metadata_arguments() -> None:
    """星云矩阵加载只按向量版本/筛选条件取数，不接收单个作品元数据。"""

    path = _module_file("mapplot")
    params = _function_param_names(path, "load_embedding_matrix")
    if params is None:
        pytest.skip("nebula.mapplot 尚无 load_embedding_matrix")
    assert params == ["cat", "cfg", "version", "filter_spec"], (
        f"load_embedding_matrix 参数异常: {params}"
    )
    assert not set(params) & set(METADATA_FIELDS)


# ──────────────────────────────────────────────────────────────────────────────
# 向量版本
# ──────────────────────────────────────────────────────────────────────────────


def _variant_version(
    config: Config,
    *,
    model: dict | None = None,
    media: dict | None = None,
    umap: dict | None = None,
    app: dict | None = None,
) -> str:
    changed = replace(
        config,
        model=replace(config.model, **model) if model else config.model,
        media=replace(config.media, **media) if media else config.media,
        umap=replace(config.umap, **umap) if umap else config.umap,
        app=replace(config.app, **app) if app else config.app,
    )
    return changed.embedding_version


def test_embedding_version_tracks_model_and_sampling(config: Config) -> None:
    """模型版本或任一采样参数变化时，向量版本必须变化（旧向量不被覆盖）。"""

    base = config.embedding_version
    variants = {
        "视频帧数": {"model": {"frames": config.model.frames + 8}},
        "片段长度": {"media": {"clip_seconds": config.media.clip_seconds / 2}},
        "片段数": {"media": {"clips_per_work": config.media.clips_per_work + 1}},
        "音频采样率": {"media": {"audio_sample_rate": 16000}},
        "模型版本标签": {"model": {"version": "peav-base-16f"}},
    }
    seen = {base}
    for label, kwargs in variants.items():
        version = _variant_version(config, **kwargs)
        assert version != base, f"{label} 变化后 embedding_version 未变: {version}"
        assert version not in seen, f"{label} 变化后与已有版本重复: {version}"
        seen.add(version)


def test_embedding_version_ignores_runtime_only_settings(config: Config) -> None:
    """UMAP / 界面 / 批大小等运行参数不改变向量版本（否则会无意义地重算向量）。"""

    base = config.embedding_version
    assert (
        _variant_version(
            config,
            umap={"random_state": config.umap.random_state + 1},
            app={"point_size": config.app.point_size + 4},
        )
        == base
    )
    assert _variant_version(config, model={"batch_size": config.model.batch_size + 1}) == base
    # device/dtype 是运行时条件；A3/ADR-0001 明确 embedding_version 不编码它们
    assert _variant_version(config, model={"device": "cpu", "dtype": "float32"}) == base
    other_device = "cuda" if config.model.device != "cuda" else "cpu"
    other_dtype = "bfloat16" if config.model.dtype != "bfloat16" else "float32"
    assert _variant_version(config, model={"device": other_device, "dtype": other_dtype}) == base


def test_embedding_version_format_and_content(config: Config) -> None:
    """版本号可读且包含模型标签与关键采样参数，便于人工核对。"""

    version = config.embedding_version
    assert slugify(config.model.version) in version
    assert f"f{config.model.frames}" in version.split("_")
    assert f"x{config.media.clips_per_work}" in version
    assert f"{config.media.audio_sample_rate // 1000}k" in version
    assert " " not in version and "/" not in version


def test_embedding_version_reproducible(config: Config) -> None:
    """同一配置多次计算版本号稳定（可复现性）。"""

    assert config.embedding_version == config.embedding_version


# ──────────────────────────────────────────────────────────────────────────────
# 状态机
# ──────────────────────────────────────────────────────────────────────────────


def test_transition_table_is_complete_and_consistent() -> None:
    """状态迁移表覆盖全部状态，目标状态合法，completed 是终态。"""

    assert set(TRANSITIONS) == set(STATUSES)
    for source, targets in TRANSITIONS.items():
        assert targets <= set(STATUSES), f"{source} 指向未知状态: {targets - set(STATUSES)}"
    assert TRANSITIONS["completed"] == frozenset(), "completed 不应有出边"
    entering_completed = {
        source for source, targets in TRANSITIONS.items() if "completed" in targets
    }
    assert entering_completed == {"encoding"}, (
        f"只有 encoding 可以进入 completed，实际: {entering_completed}"
    )
    retry_targets = TRANSITIONS["failed"] & {"pending", "downloading", "encoding"}
    assert retry_targets, "failed 应允许重试（回到 pending/downloading/encoding）"
    for source in ("pending", "downloading", "encoding"):
        assert "failed" in TRANSITIONS[source], f"{source} 应允许进入 failed"


def test_catalog_enforces_terminal_state(catalog: Catalog) -> None:
    """真实编目操作遵守状态机：completed 后不可再迁移，失败作品可重试。"""

    work_id, created = catalog.upsert_local_work("state/machine.mp4")
    assert created is True
    catalog.begin_attempt(work_id, "encoding")
    catalog.complete(work_id)
    with pytest.raises(DatabaseError):
        catalog.begin_attempt(work_id, "encoding")
    for target in ("pending", "downloading", "encoding"):
        with pytest.raises(DatabaseError):
            catalog.set_status(work_id, target)

    failed_id, _ = catalog.upsert_local_work("state/retry.mp4")
    catalog.begin_attempt(failed_id, "encoding")
    catalog.fail(failed_id, "合成失败")
    assert catalog.get_work(failed_id)["status"] == "failed"
    catalog.set_status(failed_id, "pending")
    catalog.begin_attempt(failed_id, "encoding")
    assert catalog.get_work(failed_id)["status"] == "encoding"


def test_completed_only_reopened_by_explicit_method(catalog: Catalog) -> None:
    """completed 是终态；只有显式 Catalog.reopen() 能重开（并留下 reopen 事件）。"""

    work_id, _ = catalog.upsert_local_work("state/reopen.mp4")
    catalog.begin_attempt(work_id, "encoding")
    catalog.complete(work_id)
    old = catalog.reopen(work_id, reason="模型升级，需要新向量版本")
    assert old == "completed"
    assert catalog.get_work(work_id)["status"] == "pending"
    events = catalog.get_events(work_id)
    assert any(event["event"] == "reopen" for event in events)
    # pending 再次 reopen 是幂等的，不产生新的迁移
    assert catalog.reopen(work_id) == "pending"


# ──────────────────────────────────────────────────────────────────────────────
# 去重（work_key）
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a/b.mp4", "local:a/b.mp4"),
        ("a\\b.mp4", "local:a/b.mp4"),
        ("./a/b.mp4", "local:a/b.mp4"),
        ("a//b.mp4", "local:a/b.mp4"),
    ],
)
def test_local_work_key_normalizes_relative_path(raw: str, expected: str) -> None:
    assert work_key_for_local(raw) == expected


@pytest.mark.parametrize("raw", ["C:/tmp/a.mp4", "C:\\tmp\\a.mp4", "../escape.mp4", ""])
def test_local_work_key_rejects_unsafe_paths(raw: str) -> None:
    with pytest.raises(ValueError):
        work_key_for_local(raw)


def test_local_work_dedupes_in_catalog(catalog: Catalog) -> None:
    first_id, created = catalog.upsert_local_work("series/ep01.mp4")
    second_id, created_again = catalog.upsert_local_work("series\\ep01.mp4")
    assert created is True and created_again is False
    assert first_id == second_id
    assert catalog.count_works_by_status()["pending"] == 1


def test_bilibili_work_key_dedupes_by_bvid_and_part(catalog: Catalog) -> None:
    bvid = "BV1xx411c7mD"
    key_p1 = work_key_for_bilibili(bvid, 1)
    assert key_p1 == f"bilibili:{bvid}:p1"
    assert work_key_for_bilibili(" " + bvid + " ", 1) == key_p1
    assert work_key_for_bilibili(bvid, 2) != key_p1
    assert work_key_for_bilibili(bvid, 1) == key_p1

    first_id, created = catalog.upsert_bilibili_work(bvid, title="甲")
    again_id, created_again = catalog.upsert_bilibili_work(bvid, title="乙")
    part2_id, created_p2 = catalog.upsert_bilibili_work(bvid, part=2)
    assert created is True and created_again is False and created_p2 is True
    assert first_id == again_id != part2_id
    assert catalog.count_works_by_status()["pending"] == 2

    with pytest.raises(DatabaseError):
        work_key_for_bilibili(bvid, 0)
    with pytest.raises(DatabaseError):
        work_key_for_bilibili("", 1)


# ──────────────────────────────────────────────────────────────────────────────
# DB v2（迭代 A）：身份关联 / 批次计时 / 筛选条件落库
# ──────────────────────────────────────────────────────────────────────────────


def test_schema_v3_tables_and_columns(catalog: Catalog) -> None:
    """v3 schema：v2 表/列保留，并新增表征/序列表与 v0.4 列；版本号必须为 3。"""

    assert SCHEMA_VERSION == "3"
    tables = {
        str(row[0])
        for row in catalog.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {
        "work_links",
        "pipeline_runs",
        "work_timings",
        "representations",
        "representation_vectors",
        "sequence_states",
    } <= tables
    map_columns = {
        str(row[1]) for row in catalog.conn.execute("PRAGMA table_info(map_runs)").fetchall()
    }
    assert {"filter_json", "representation_id"} <= map_columns
    link_columns = {
        str(row[1]) for row in catalog.conn.execute("PRAGMA table_info(work_links)").fetchall()
    }
    assert {"canonical_work_id", "duplicate_work_id", "method"} <= link_columns
    clip_columns = {
        str(row[1]) for row in catalog.conn.execute("PRAGMA table_info(clips)").fetchall()
    }
    assert {"valid_duration_s", "weight"} <= clip_columns
    timing_columns = {
        str(row[1]) for row in catalog.conn.execute("PRAGMA table_info(work_timings)").fetchall()
    }
    assert "events_s" in timing_columns


def test_work_links_single_level_and_unique(catalog: Catalog) -> None:
    """一个 duplicate 只能属于一个 canonical；canonical 不能是别人的 duplicate。"""

    canonical, _ = catalog.upsert_bilibili_work("BV1xx411c7m1")
    duplicate_a, _ = catalog.upsert_bilibili_work("BV1xx411c7m2")
    duplicate_b, _ = catalog.upsert_bilibili_work("BV1xx411c7m3")

    catalog.link_works(
        canonical, duplicate_a, method="title+duration", confidence=0.99, note="测试"
    )
    with pytest.raises(DatabaseError):
        catalog.link_works(duplicate_b, duplicate_a, method="title+duration")
    with pytest.raises(DatabaseError):
        catalog.link_works(duplicate_a, duplicate_b, method="title+duration")
    with pytest.raises(DatabaseError):
        catalog.link_works(duplicate_b, duplicate_b, method="title+duration")
    with pytest.raises(DatabaseError):
        catalog.link_works(canonical, duplicate_b, method="")

    links = catalog.get_work_links()
    assert len(links) == 1
    assert links[0]["canonical_work_id"] == canonical
    assert links[0]["duplicate_work_id"] == duplicate_a
    assert catalog.get_canonical_work_id(duplicate_a) == canonical
    assert catalog.get_canonical_work_id(canonical) == canonical
    assert catalog.get_duplicate_work_ids() == {duplicate_a}
    linked = catalog.get_linked_duplicates(canonical)
    assert [int(row["id"]) for row in linked] == [duplicate_a]
    assert linked[0]["link_method"] == "title+duration"
    assert linked[0]["link_confidence"] == pytest.approx(0.99)
    assert catalog.get_work_link(duplicate_a)["canonical_work_id"] == canonical
    with pytest.raises(DatabaseError):
        catalog.unlink_work(canonical)  # canonical 本身没有关联记录
    assert catalog.unlink_work(duplicate_a, reason="撤销") == canonical
    assert catalog.get_duplicate_work_ids() == set()
    assert catalog.get_work_links() == []


def test_duplicates_excluded_from_default_queries_but_data_kept(catalog: Catalog) -> None:
    """duplicate 默认不进入榜单/地图候选查询，但其作品记录、向量与文件登记保留。"""

    catalog.register_embedding_version(
        "v-link",
        model_checkpoint="synthetic",
        model_revision=None,
        frames=16,
        clip_seconds=8.0,
        clips_per_work=5,
        audio_sample_rate=48000,
        params={},
    )
    canonical, _ = catalog.upsert_bilibili_work("BV1xx411c7m1", title="同一作品")
    duplicate, _ = catalog.upsert_local_work("local/同一作品.mp4")
    for work_id in (canonical, duplicate):
        catalog.register_embedding(
            work_id,
            version="v-link",
            kind="work_joint",
            path=f"embeddings/v-link/{work_id:06d}/work_joint.npy",
            dim=8,
        )
    list_id = catalog.ensure_list(
        "votes_2025", name="2025年音MAD选票", kind="votes", vote_year=2025
    )
    catalog.add_appearance(canonical, list_id, rank=1)
    catalog.add_appearance(duplicate, list_id, rank=2)
    catalog.link_works(canonical, duplicate, method="title+duration", confidence=0.99)

    assert catalog.work_ids_for_list("votes_2025") == {canonical}
    assert catalog.work_ids_for_list("votes_2025", only_canonical=False) == {
        canonical,
        duplicate,
    }
    default_rows = catalog.list_works_for_list("votes_2025")
    assert [int(row["id"]) for row in default_rows] == [canonical]
    full_rows = catalog.list_works_for_list("votes_2025", only_canonical=False)
    assert {int(row["id"]) for row in full_rows} == {canonical, duplicate}
    # duplicate 的向量登记与作品记录必须保留（只是默认不出现在星云/榜单视图里）
    assert catalog.has_embedding(duplicate, version="v-link")
    assert catalog.has_embedding(canonical, version="v-link")


def test_map_run_filter_json_roundtrip(catalog: Catalog) -> None:
    """筛选星云的 filter_spec 必须原样落库；无筛选时为 None。"""

    catalog.register_embedding_version(
        "v1",
        model_checkpoint="synthetic",
        model_revision=None,
        frames=16,
        clip_seconds=8.0,
        clips_per_work=5,
        audio_sample_rate=48000,
        params={},
    )
    work_id, _ = catalog.upsert_local_work("a.mp4")
    catalog.create_map_run(
        "map_v001",
        embedding_version="v1",
        params={"n_neighbors": 15},
        coords=[(work_id, 0.0, 0.0, 0.0, 0.0, 0.0)],
    )
    assert catalog.get_map("map_v001")["filter_json"] is None

    spec = {"dataset": "bilibili", "vote_year": 2025, "include_duplicates": False}
    catalog.create_map_run(
        "map_v002",
        embedding_version="v1",
        params={"n_neighbors": 8},
        coords=[(work_id, 1.0, 1.0, 1.0, 1.0, 1.0)],
        filter_spec=spec,
    )
    row = catalog.get_map("map_v002")
    assert json.loads(row["filter_json"]) == spec
    latest = catalog.latest_map()
    assert latest["version"] == "map_v002"
    assert json.loads(latest["filter_json"]) == spec
    assert all("filter_json" in map_row for map_row in catalog.list_maps())


def test_pipeline_runs_and_timings_semantics(catalog: Catalog) -> None:
    """批次计时：负耗时/负计数被拒绝，写读一致，批次完成信息可查。"""

    work_id, _ = catalog.upsert_local_work("a.mp4")
    run_id = catalog.start_pipeline_run(embedding_version="v1", params={"note": "审计"})
    assert isinstance(run_id, int) and run_id >= 1

    catalog.record_work_timing(
        None, work_id, embedding_version="v1", probe_s=0.1, total_s=0.2
    )
    catalog.record_work_timing(
        run_id, work_id, embedding_version="v1", extract_s=0.3, total_s=0.4
    )
    timings = catalog.list_work_timings(run_id=run_id)
    assert len(timings) == 1
    assert timings[0]["extract_s"] == pytest.approx(0.3)
    assert len(catalog.list_work_timings(embedding_version="v1")) == 2

    catalog.finish_pipeline_run(
        run_id, n_selected=1, n_completed=1, n_failed=0, total_s=1.5, model_load_s=0.2
    )
    runs = catalog.list_pipeline_runs(limit=5)
    assert runs[0]["id"] == run_id
    assert runs[0]["finished_at"]
    assert runs[0]["n_completed"] == 1
    assert runs[0]["total_s"] == pytest.approx(1.5)

    for field in (
        "download_s",
        "probe_s",
        "extract_s",
        "preprocess_s",
        "forward_s",
        "save_s",
        "total_s",
    ):
        with pytest.raises(DatabaseError):
            catalog.record_work_timing(
                run_id, work_id, embedding_version="v1", **{field: -0.01}
            )
    with pytest.raises(DatabaseError):
        catalog.finish_pipeline_run(99999, n_selected=1, n_completed=1, n_failed=0, total_s=1)
    with pytest.raises(DatabaseError):
        catalog.list_pipeline_runs(limit=0)

    stats = catalog.stats()
    assert stats["links"] == 0
    assert stats["pipeline_runs"] == 1
    assert stats["timings"] >= 1


# ──────────────────────────────────────────────────────────────────────────────
# v0.4：全片序列 / 表征版本 / 空间隔离
# ──────────────────────────────────────────────────────────────────────────────


def _v04_register_version(
    catalog: Catalog,
    version: str,
    *,
    clips_per_work: int = 5,
    clip_seconds: float = 8.0,
) -> None:
    catalog.register_embedding_version(
        version,
        model_checkpoint="synthetic",
        model_revision=None,
        frames=16,
        clip_seconds=clip_seconds,
        clips_per_work=clips_per_work,
        audio_sample_rate=48000,
        params={"test": "v0.4-invariants"},
    )


def _v04_unit(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array / float(np.linalg.norm(array))


def _v04_local_work(catalog: Catalog, name: str) -> int:
    work_id, _ = catalog.upsert_local_work(f"v04/{name}.mp4", title=name)
    return int(work_id)


def _v04_save_embedding(
    catalog: Catalog,
    config: Config,
    work_id: int,
    version: str,
    vector: np.ndarray,
) -> None:
    path = config.paths.embeddings / version / f"{work_id:06d}" / "work_joint.npy"
    save_vector(path, vector)
    catalog.register_embedding(
        work_id,
        version=version,
        kind="work_joint",
        path=config.data_relpath(path),
        dim=int(vector.size),
    )


def _v04_create_representation(
    catalog: Catalog,
    config: Config,
    *,
    name: str,
    sequence_version: str,
    vectors: dict[int, np.ndarray],
) -> int:
    rep_id = catalog.register_representation(
        name,
        sequence_version=sequence_version,
        kind="full_mean",
        params={"aggregator": "mean", "test": "v0.4-invariants"},
    )
    rep = catalog.get_representation(rep_id)
    assert rep is not None
    for work_id, vector in vectors.items():
        save_representation_vectors(catalog, config, rep, work_id, {"joint": vector})
    first = next(iter(vectors.values()))
    catalog.set_representation_status(rep_id, "ready", dim=int(first.size))
    return int(rep_id)


def test_sampling_mode_switch_changes_active_version_not_legacy_formula(config: Config) -> None:
    """sampling.mode 只切换“当前激活序列版本”；legacy 公式必须与旧版完全一致。"""

    expected_legacy = (
        f"{slugify(config.model.version)}"
        f"__f{config.model.frames}"
        f"_clip{config.media.clip_seconds:g}s"
        f"_x{config.media.clips_per_work}"
        f"_{config.media.audio_sample_rate // 1000}k"
    )
    if config.media.clip_height:
        expected_legacy += f"_h{int(config.media.clip_height)}"
    assert config.sampling.mode == "legacy"
    assert config.legacy_embedding_version == expected_legacy
    assert config.embedding_version == expected_legacy

    full = replace(config, sampling=replace(config.sampling, mode="full_coverage"))
    assert full.legacy_embedding_version == expected_legacy
    assert full.embedding_version == full.full_sequence_version
    assert full.embedding_version != config.embedding_version
    assert "full" in full.embedding_version
    assert f"_ov{full.sampling.overlap_seconds:g}" in full.embedding_version
    assert f"x{config.media.clips_per_work}" not in full.full_sequence_version

    # 改变 overlap 只会生成新的 full 序列版本，不改变 legacy 版本号
    shifted = replace(full, sampling=replace(full.sampling, overlap_seconds=full.sampling.overlap_seconds + 0.5))
    assert shifted.full_sequence_version != full.full_sequence_version
    assert shifted.legacy_embedding_version == expected_legacy


def test_full_sequence_version_format_and_clip_height(config: Config) -> None:
    """full 版本号含 full/ov 与可选 _h 标记，且与 legacy 版本严格不同。"""

    full = replace(config, sampling=replace(config.sampling, mode="full_coverage"))
    assert full.full_sequence_version != full.legacy_embedding_version
    assert f"_full{config.media.clip_seconds:g}s" in full.full_sequence_version
    assert f"_ov{config.sampling.overlap_seconds:g}" in full.full_sequence_version
    assert " " not in full.full_sequence_version and "/" not in full.full_sequence_version

    tall = replace(config, media=replace(config.media, clip_height=336))
    tall_full = replace(tall, sampling=replace(tall.sampling, mode="full_coverage"))
    assert tall_full.full_sequence_version.endswith("_h336")
    assert "_h336" in tall_full.legacy_embedding_version

    plain = replace(config, media=replace(config.media, clip_height=None))
    plain_full = replace(plain, sampling=replace(plain.sampling, mode="full_coverage"))
    assert "_h" not in plain_full.full_sequence_version


def test_representations_isolated_by_name_sequence_and_kind(
    catalog: Catalog, config: Config
) -> None:
    """不同表征/不同序列的向量矩阵彼此隔离，legacy 与派生表征不混用。"""

    _v04_register_version(catalog, "v04-a")
    _v04_register_version(catalog, "v04-b")
    first = _v04_local_work(catalog, "rep-a1")
    second = _v04_local_work(catalog, "rep-a2")
    third = _v04_local_work(catalog, "rep-b1")
    fourth = _v04_local_work(catalog, "rep-a3-only-legacy")

    vectors_a = {
        first: _v04_unit([1.0, 0.0, 0.0, 0.0]),
        second: _v04_unit([0.0, 1.0, 0.0, 0.0]),
    }
    vectors_b = {third: _v04_unit([1.0, 0.0, 0.0])}
    for work_id, vector in vectors_a.items():
        _v04_save_embedding(catalog, config, work_id, "v04-a", vector)
    _v04_save_embedding(catalog, config, fourth, "v04-a", _v04_unit([0.0, 0.0, 1.0, 0.0]))

    rep_a_id = _v04_create_representation(
        catalog, config, name="full_mean_v1", sequence_version="v04-a", vectors=vectors_a
    )
    rep_b_id = _v04_create_representation(
        catalog, config, name="full_mean_v1", sequence_version="v04-b", vectors=vectors_b
    )
    rep_a = catalog.get_representation(rep_a_id)
    rep_b = catalog.get_representation(rep_b_id)
    assert rep_a is not None and rep_b is not None

    ids_a, matrix_a = load_representation_matrix(catalog, config, rep_a)
    ids_b, matrix_b = load_representation_matrix(catalog, config, rep_b)
    assert set(ids_a) == {first, second}
    assert set(ids_b) == {third}
    assert set(ids_a).isdisjoint(ids_b)
    assert matrix_a.shape == (2, 4)
    assert matrix_b.shape == (1, 3)

    # legacy 表征只读 embeddings 表，不包含任何代表征向量
    legacy_id = catalog.ensure_legacy_representation("v04-a")
    legacy_rep = catalog.get_representation(legacy_id)
    assert legacy_rep is not None
    legacy_ids, legacy_matrix = load_representation_matrix(catalog, config, legacy_rep)
    assert set(legacy_ids) == {first, second, fourth}
    assert legacy_matrix.shape[1] == 4
    assert third not in set(legacy_ids)

    # 同名 + 同序列必须解析到同一条记录；不同序列不能串号
    assert resolve_representation(
        catalog, config, name="full_mean_v1", sequence_version="v04-a"
    )["id"] == rep_a_id
    assert resolve_representation(
        catalog, config, name="full_mean_v1", sequence_version="v04-b"
    )["id"] == rep_b_id
    with pytest.raises(DatabaseError):
        catalog.register_representation(
            "full_mean_v1",
            sequence_version="v04-a",
            kind="full_mean",
            params={"aggregator": "sum"},
        )
    with pytest.raises(NebulaError):
        resolve_representation(catalog, config, name="not_built_representation")


def test_maps_bind_representation_and_never_overwrite_old_coords(catalog: Catalog) -> None:
    """地图必须绑定“同序列”的表征；新地图不覆盖旧地图坐标。"""

    _v04_register_version(catalog, "v04-map")
    _v04_register_version(catalog, "v04-other")
    first = _v04_local_work(catalog, "map1")
    second = _v04_local_work(catalog, "map2")
    # 地图绑定测试不需要真实向量文件；直接登记 ready 表征
    rep_id = catalog.register_representation(
        "full_mean_v1",
        sequence_version="v04-map",
        kind="full_mean",
        params={"test": "map-binding"},
    )
    catalog.set_representation_status(rep_id, "ready", dim=4)

    catalog.create_map_run(
        "map_v001",
        embedding_version="v04-map",
        params={"note": "旧地图未绑定表征"},
        coords=[(first, 1.0, 2.0, 3.0, 4.0, 5.0), (second, -1.0, -2.0, -3.0, -4.0, -5.0)],
    )
    catalog.create_map_run(
        "map_v002",
        embedding_version="v04-map",
        params={"note": "绑定表征"},
        coords=[(first, 9.0, 8.0, 7.0, 6.0, 5.0)],
        representation_id=rep_id,
    )
    assert catalog.get_map("map_v001")["representation_id"] is None
    assert catalog.get_map("map_v002")["representation_id"] == rep_id
    assert catalog.latest_map()["version"] == "map_v002"

    # 表征序列与 embedding_version 不一致时必须拒绝建图
    with pytest.raises(DatabaseError):
        catalog.create_map_run(
            "map_v003",
            embedding_version="v04-other",
            params={},
            coords=[(first, 0.0, 0.0, 0.0, 0.0, 0.0)],
            representation_id=rep_id,
        )
    # 旧地图版本不可覆盖（重复版本号被拒绝，坐标保持第一次写入的值）
    with pytest.raises(DatabaseError):
        catalog.create_map_run(
            "map_v001",
            embedding_version="v04-map",
            params={},
            coords=[(first, 100.0, 100.0, 100.0, 100.0, 100.0)],
        )
    old_rows = {int(row["id"]): row for row in catalog.load_map_rows("map_v001")}
    assert old_rows[first]["x2"] == pytest.approx(1.0)
    assert old_rows[second]["x2"] == pytest.approx(-1.0)
    new_rows = catalog.load_map_rows("map_v002")
    assert new_rows[0]["x2"] == pytest.approx(9.0)


def test_catalog_initialize_and_legacy_registration_idempotent(config: Config) -> None:
    """重复 initialize/登记 legacy 表征不产生重复行；clips 5 元组与序列状态可恢复。"""

    catalog = Catalog(config.paths.db_path)
    try:
        catalog.initialize()
        _v04_register_version(catalog, "v04-init")
        work_id = _v04_local_work(catalog, "init")
        legacy_id = catalog.ensure_legacy_representation("v04-init")
        assert catalog.ensure_legacy_representation("v04-init") == legacy_id
        catalog.replace_clips(
            work_id,
            version="v04-init",
            clips=[(0, 0.0, 8.0, 8.0, 0.5), (1, 8.0, 8.0, 8.0, 0.5)],
        )
        catalog.upsert_sequence_state(
            work_id,
            version="v04-init",
            duration_s=16.0,
            n_clips=2,
            n_clips_done=1,
            partial_relpath="temp/v04-init/000001",
            status="in_progress",
        )
        before = (
            len(catalog.list_representations()),
            len(catalog.list_maps()),
            len(catalog.get_clips(work_id, version="v04-init")),
        )
        catalog.initialize()
        catalog.initialize()
        assert (
            len(catalog.list_representations()),
            len(catalog.list_maps()),
            len(catalog.get_clips(work_id, version="v04-init")),
        ) == before
        assert catalog.ensure_legacy_representation("v04-init") == legacy_id
        clips = catalog.get_clips(work_id, version="v04-init")
        assert clips[0]["valid_duration_s"] == pytest.approx(8.0)
        assert clips[0]["weight"] == pytest.approx(0.5)
        state = catalog.get_sequence_state(work_id, version="v04-init")
        assert state is not None
        assert state["n_clips"] == 2 and state["n_clips_done"] == 1
        catalog.upsert_sequence_state(
            work_id,
            version="v04-init",
            duration_s=16.0,
            n_clips=2,
            n_clips_done=2,
            partial_relpath=None,
            status="assembled",
        )
        assembled = catalog.get_sequence_state(work_id, version="v04-init")
        assert assembled is not None and assembled["status"] == "assembled"
    finally:
        catalog.close()


def test_compare_spaces_tool_import_boundary(repo_root: Path) -> None:
    """邻域迁移工具必须离线/纯数值：不依赖模型、UMAP、pandas 或界面库。"""

    tool = repo_root / "tools" / "compare_spaces.py"
    assert tool.is_file(), "tools/compare_spaces.py 尚未交付"
    _assert_no_import(
        tool,
        ("torch", "transformers", "umap", "streamlit", "plotly", "pandas", "nebula.embed"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 采样规则
# ──────────────────────────────────────────────────────────────────────────────


def test_sampling_never_produces_negative_or_out_of_bounds_clips() -> None:
    """对随机时长做性质检查：片段非负、非空、不越界、不重叠、索引连续。"""

    rng = random.Random(20250101)
    durations = [1.0, 1.001, 7.999, 8.0, 8.001, 16.0, 16.001, 40.0]
    durations += [rng.uniform(1.0, 600.0) for _ in range(120)]
    durations += [10 ** rng.uniform(0, 3) for _ in range(40)]
    for duration in durations:
        plans = plan_clips(duration)
        assert plans, f"时长 {duration} 没有生成片段"
        assert len(plans) <= 5, f"时长 {duration} 片段数超过上限"
        assert [plan.index for plan in plans] == list(range(len(plans)))
        # 短视频的片段长度会四舍五入到毫秒，最多可上浮 0.5ms；长视频要求严格不越界。
        tolerance = 1.5e-3 if duration < 8.0 else 1e-9
        previous_end = -1.0
        for plan in plans:
            assert plan.duration_s > 0, f"时长 {duration} 出现非正片段长度"
            assert plan.start_s >= 0, f"时长 {duration} 出现负起点"
            assert plan.start_s + plan.duration_s <= duration + tolerance, (
                f"时长 {duration} 片段越界: {plan}"
            )
            assert plan.start_s >= previous_end - 1e-9, f"时长 {duration} 片段重叠"
            previous_end = plan.start_s + plan.duration_s
        assert plan_clips(duration) == plans, f"时长 {duration} 采样不确定"


def test_sampling_short_video_uses_whole_video() -> None:
    """短于目标片段长度的视频整段作为唯一片段，从 0 开始。"""

    for duration in (1.0, 3.0, 7.5, 7.999):
        plans = plan_clips(duration)
        assert len(plans) == 1
        assert plans[0].start_s == 0.0
        assert plans[0].duration_s == pytest.approx(duration, abs=1e-3)


def test_sampling_rejects_invalid_durations() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(SamplingError):
            plan_clips(bad)
    with pytest.raises(SamplingError):
        plan_clips(0.5, min_duration_seconds=1.0)


# ──────────────────────────────────────────────────────────────────────────────
# 向量种类
# ──────────────────────────────────────────────────────────────────────────────


def test_embedding_kinds_partition() -> None:
    """作品级 / 片段级向量种类互斥且合起来覆盖全部登记种类。"""

    assert set(WORK_VECTOR_KINDS).isdisjoint(CLIP_VECTOR_KINDS)
    assert tuple(WORK_VECTOR_KINDS) + tuple(CLIP_VECTOR_KINDS) == tuple(EMBEDDING_KINDS)


# ──────────────────────────────────────────────────────────────────────────────
# .gitignore 与 CLI 入口
# ──────────────────────────────────────────────────────────────────────────────


def test_gitignore_mentions_runtime_artifacts(repo_root: Path) -> None:
    text = (repo_root / ".gitignore").read_text(encoding="utf-8")
    required = (
        ".venv",
        "data/catalog.sqlite",
        "data/embeddings",
        "data/maps",
        "temp/",
        "logs/",
        "input/local",
        "*.mp4",
        "*.npy",
    )
    missing = [token for token in required if token not in text]
    assert not missing, f".gitignore 缺少运行时产物规则: {missing}"


def test_gitignore_actually_ignores_paths(repo_root: Path) -> None:
    """用 git check-ignore 做语义级验证（不是只看文本模式）。"""

    git = shutil.which("git")
    if git is None or not (repo_root / ".git").exists():
        pytest.skip("当前环境没有可用的 git 仓库，跳过语义检查")
    runtime_paths = (
        ".venv/Scripts/python.exe",
        "data/catalog.sqlite",
        "data/embeddings/test-version/000001/work_joint.npy",
        "data/maps/map_v001/coords.parquet",
        "temp/clip_0.mp4",
        "logs/pipeline.log",
        "input/local/demo.mp4",
        "tests/fixtures/out/manifest.json",
    )
    for rel_path in runtime_paths:
        proc = subprocess.run(
            [git, "check-ignore", "--no-index", "-q", "--", rel_path],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        assert proc.returncode == 0, f"git 未忽略运行时产物: {rel_path}"


@pytest.mark.parametrize(
    ("script_name", "args"),
    [
        ("ingest.py", ()),
        ("pipeline.py", ()),
        ("map.py", ()),
        ("map.py", ("variants",)),
        ("ingest.py", ("link-local",)),
        ("tools/clip_stability.py", ()),
        ("tools/compare_spaces.py", ()),
        ("qa.py", ()),
    ],
)
def test_cli_help_runs(repo_root: Path, script_name: str, args: tuple[str, ...]) -> None:
    """顶层 CLI 与 v0.2 子命令必须能被 --help 安全调用（不加载模型、不触网、不写数据）。"""

    script = repo_root / script_name
    if not script.is_file():
        pytest.skip(f"入口脚本 {script_name} 尚未交付")
    label = " ".join((script_name, *args))
    proc = subprocess.run(
        [sys.executable, str(script), *args, "--help"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, f"{label} --help 退出码 {proc.returncode}: {combined[-800:]}"
    assert "usage" in combined.lower(), f"{label} --help 未输出 usage"
    assert "traceback" not in combined.lower(), f"{label} --help 出现异常堆栈"
