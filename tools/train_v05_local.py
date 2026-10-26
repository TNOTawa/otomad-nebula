"""v0.5-beta 局部帧级时序模型训练/评估 CLI（temporal-only / cross-attn）。

真实运行（需 t3 的 data/v05_protocol/v2 已就绪，GPU 由 Lead 跑）::

    .\\.venv\\Scripts\\python.exe tools/train_v05_local.py --arch both --device cuda
    .\\.venv\\Scripts\\python.exe tools/train_v05_local.py --arch cross --seeds 42 43 44 --epochs 20

CPU smoke（合成协议 + 合成帧，不读真实数据、不联网）::

    .\\.venv\\Scripts\\python.exe tools/train_v05_local.py --smoke --arch both --epochs 1 --seeds 42

说明：
- 直接复用 data/visual_dense/<CLIP 8fps>/<work_id>/dense_visual.npz 与
  data/audio_dense/<audio_version>/<work_id>/audio.npz，不重跑编码；
- 候选/切分完全读取 t3 协议 manifest（work_id/anchor_bin/candidate_bins/labels/
  split/periodic），不在本工具里重造协议；train/val/test works 必须不重叠；
- 仅 InfoNCE 一个目标；两个 arch 共享同一协议与 DataLoader 配置；
- 产物：data/models/v05_local_<arch>/seed_<seed>/checkpoint.pt（含 config/state_dict/
  metrics）与 logs/v05_local_<arch>.json（每 seed 指标 + 均值/方差 + 协议 baselines
  引用）；不注册 DB/地图、不启动全曲模型。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nebula.config import load_config  # noqa: E402
from nebula.errors import NebulaError  # noqa: E402
from nebula.log import configure_logging  # noqa: E402
from nebula.util import now_utc_iso  # noqa: E402
from nebula.v05_local import (  # noqa: E402
    ARCHS,
    DEFAULT_ABLATIONS,
    DEFAULT_AUDIO_HZ,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BOOTSTRAP,
    DEFAULT_CONTEXT_SECONDS,
    DEFAULT_CROSS_LAYERS,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_HEADS,
    DEFAULT_HIDDEN,
    DEFAULT_K_RETRIEVAL,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    DEFAULT_SEEDS,
    DEFAULT_SELF_LAYERS,
    DEFAULT_TEMPERATURE,
    DEFAULT_WEIGHT_DECAY,
    DIAGNOSTIC_CRITERIA,
    FrameDataStore,
    LocalModelConfig,
    LocalPairDataset,
    LocalTrainConfig,
    V05LocalError,
    build_local_model,
    build_synthetic_batches,
    collate_local_sets,
    evaluate_local,
    load_checkpoint,
    load_protocol,
    make_synthetic_protocol,
    save_checkpoint,
    summarize_local_diagnostics,
    train_local_model,
)

LOGGER = logging.getLogger("v05_local")

DEFAULT_REFERENCE_BASELINES = {
    "source": "t3 logs/v05_protocol_v2_metrics.json（同一 test 协议；未重算）",
    "main_k6_top1": {
        "random": 0.166667,
        "frozen_projection": 0.162728,
        "simple_sync": 0.175496,
        "v05_beta_42": 0.179571,
        "v05_beta_43": 0.179027,
        "v05_beta_44": 0.179842,
    },
    "retrieval_k32_top1": {"random": 0.03125},
}


def _resolve_config_path(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    env_value = os.environ.get("NEBULA_CONFIG")
    if env_value:
        return Path(env_value).expanduser()
    return Path.cwd() / "config.yaml"


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise V05LocalError("--device cuda 但当前环境没有可用 CUDA")
    return requested


def _build_model_config(args: argparse.Namespace, *, visual_dim: int, audio_dim: int) -> LocalModelConfig:
    return LocalModelConfig(
        hidden=int(args.hidden),
        heads=int(args.heads),
        self_layers=int(args.self_layers),
        cross_layers=int(args.cross_layers),
        dropout=float(args.dropout),
        context_seconds=float(args.context_seconds),
        audio_hz=float(args.audio_hz),
        visual_dim=int(visual_dim),
        audio_dim=int(audio_dim),
        temperature=float(args.temperature),
    )


def _build_train_config(args: argparse.Namespace, model_config: LocalModelConfig, arch: str) -> LocalTrainConfig:
    return LocalTrainConfig(
        arch=arch,
        seeds=tuple(int(value) for value in args.seeds),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        patience=int(args.patience),
        bootstrap=int(args.bootstrap),
        k_retrieval=int(args.k_retrieval),
        device=str(args.device),
        num_workers=int(args.num_workers),
        model=model_config,
    ).validate()


def _split_loaders(
    protocol: Any,
    store: Any,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    limit: int | None,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    records = protocol.split_records()
    loaders: dict[str, Any] = {}
    for name, split_records in records.items():
        items = list(split_records)
        if limit is not None:
            items = items[: int(limit)]
        if not items:
            raise V05LocalError(
                f"协议 split={name} 没有记录；请检查 t3 协议（train/val/test 都不能为空）"
            )
        dataset = LocalPairDataset(items, store, bin_seconds=protocol.bin_seconds)
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        loaders[name] = DataLoader(
            dataset,
            batch_size=max(1, int(batch_size)),
            shuffle=(name == "train"),
            num_workers=max(0, int(num_workers)),
            collate_fn=collate_local_sets,
            generator=generator if name == "train" else None,
        )
    retrieval_records = tuple(getattr(protocol, "retrieval_records", ()) or ())
    if limit is not None:
        retrieval_records = retrieval_records[: int(limit)]
    if retrieval_records:
        retrieval_dataset = LocalPairDataset(
            retrieval_records, store, bin_seconds=protocol.bin_seconds
        )
        loaders["retrieval"] = DataLoader(
            retrieval_dataset,
            batch_size=max(1, int(batch_size)),
            shuffle=False,
            num_workers=max(0, int(num_workers)),
            collate_fn=collate_local_sets,
        )
    return loaders


def _run_single(
    *,
    arch: str,
    seed: int,
    protocol: Any,
    loaders: dict[str, Any],
    train_config: LocalTrainConfig,
    device: str,
    out_dir: Path,
    synthetic: bool,
    reference_baselines: Any = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = build_local_model(arch, train_config.model)
    result = train_local_model(
        model,
        loaders["train"],
        loaders["val"],
        train_config,
        device=device,
        seed=int(seed),
    )
    trained = result["model"]
    test_metrics = evaluate_local(
        trained,
        loaders["test"],
        device=device,
        k_retrieval=int(train_config.k_retrieval),
        bootstrap=int(train_config.bootstrap),
        bootstrap_seed=int(seed) + 1000,
    )
    retrieval_metrics = None
    if "retrieval" in loaders:
        retrieval_metrics = evaluate_local(
            trained,
            loaders["retrieval"],
            device=device,
            k_retrieval=32,
            bootstrap=int(train_config.bootstrap),
            bootstrap_seed=int(seed) + 2000,
            ablations=("normal",),
        )
    elapsed = time.perf_counter() - started
    checkpoint_dir = out_dir / f"v05_local_{arch}" / f"seed_{int(seed)}"
    checkpoint_path = save_checkpoint(
        checkpoint_dir / "checkpoint.pt",
        model=trained,
        arch=arch,
        model_config=train_config.model,
        train_config=train_config,
        seed=int(seed),
        metrics={
            "test": test_metrics,
            "best_val_top1": result["best_val_top1"],
            "history": result["history"],
        },
        extra={
            "protocol_version": protocol.metadata.get("protocol_version"),
            "synthetic": bool(synthetic),
            "elapsed_s": round(float(elapsed), 3),
        },
    )
    metrics = {
        "arch": arch,
        "seed": int(seed),
        "model_config": train_config.model.to_dict(),
        "train_config": train_config.to_dict(),
        "protocol_version": protocol.metadata.get("protocol_version"),
        "protocol_bin_seconds": float(protocol.bin_seconds),
        "split_work_ids": {
            name: list(ids) for name, ids in protocol.split_work_ids().items()
        },
        "split_counts": {
            name: len(records)
            for name, records in protocol.split_records().items()
        },
        "history": result["history"],
        "epochs_run": int(result["epochs_run"]),
        "best_val_top1": result["best_val_top1"],
        "test": test_metrics,
        "test_retrieval": retrieval_metrics,
        "protocol_main_k": protocol.metadata.get("main_k"),
        "protocol_retrieval_k": protocol.metadata.get("retrieval_k"),
        "protocol_anchors_counts": protocol.metadata.get("anchors_counts"),
        "reference_baselines": (
            reference_baselines
            or protocol.metadata.get("baselines")
            or protocol.metadata.get("reference_baselines")
            or protocol.metadata.get("baseline_results")
        ),
        "synthetic": bool(synthetic),
        "elapsed_s": round(float(elapsed), 3),
        "checkpoint": str(checkpoint_path),
        "created_at": now_utc_iso(),
    }
    checkpoint_dir.joinpath("metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def _aggregate(metrics_list: list[dict[str, Any]], arch: str) -> dict[str, Any]:
    def _values(key: str) -> list[float]:
        values: list[float] = []
        for item in metrics_list:
            value = item["test"].get(key)
            if value is not None:
                values.append(float(value))
        return values

    def _stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "std": None, "n": 0}
        import numpy as np

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "n": len(values),
        }

    retrieval_values: list[float] = []
    for item in metrics_list:
        retrieval = item.get("test_retrieval")
        if retrieval and retrieval.get("top1") is not None:
            retrieval_values.append(float(retrieval["top1"]))
    return {
        "arch": arch,
        "seeds": [int(item["seed"]) for item in metrics_list],
        "test_top1": _stats(_values("top1")),
        "test_top1_k32": _stats(_values("top1_k32")),
        "test_retrieval_top1": _stats(retrieval_values),
        "reference_baselines": metrics_list[0].get("reference_baselines") if metrics_list else None,
        "seeds_metrics": metrics_list,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_reference_baselines(logs_dir: Path | None) -> dict[str, Any]:
    """引用 t3 同协议基线（若有 logs/v05_protocol_v2_metrics.json），否则用默认值。"""

    if logs_dir is not None:
        path = Path(logs_dir) / "v05_protocol_v2_metrics.json"
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                payload = None
            results = payload.get("results") if isinstance(payload, dict) else None
            if isinstance(results, dict):
                main_values: dict[str, float] = {}
                retrieval_values: dict[str, float] = {}
                for model, model_payload in results.items():
                    variants = (
                        model_payload.get("variants")
                        if isinstance(model_payload, dict)
                        else None
                    )
                    if not isinstance(variants, dict):
                        continue
                    main_variant = variants.get("main_k6")
                    if isinstance(main_variant, dict) and main_variant.get("top1") is not None:
                        main_values[str(model)] = round(float(main_variant["top1"]), 6)
                    retrieval_variant = variants.get("retrieval_k32")
                    if (
                        isinstance(retrieval_variant, dict)
                        and retrieval_variant.get("top1") is not None
                    ):
                        retrieval_values[str(model)] = round(
                            float(retrieval_variant["top1"]), 6
                        )
                if main_values or retrieval_values:
                    return {
                        "source": str(path),
                        "main_k6_top1": main_values,
                        "retrieval_k32_top1": retrieval_values,
                    }
    return json.loads(json.dumps(DEFAULT_REFERENCE_BASELINES, ensure_ascii=False))


def _average_work_metrics(
    variants: list[Mapping[str, Any]],
) -> tuple[dict[str, float], float | None]:
    per_work_values: dict[str, list[float]] = {}
    for variant in variants:
        for work_id, value in (variant.get("per_work_accuracy") or {}).items():
            if value is not None:
                per_work_values.setdefault(str(work_id), []).append(float(value))
    averaged = {
        work_id: float(np.mean(values))
        for work_id, values in per_work_values.items()
        if values
    }
    macro = float(np.mean(list(averaged.values()))) if averaged else None
    return averaged, macro


def _load_diagnostic_reference(logs_dir: Path) -> dict[str, Any]:
    """从协议完整 test 日志构造 simple_sync / old_beta 的 reference（含 per-work）。"""

    path = Path(logs_dir) / "v05_protocol_v2_metrics.json"
    if not path.is_file():
        raise V05LocalError(f"诊断需要完整协议基线日志: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise V05LocalError(f"读取协议基线日志失败（{path}）：{exc}") from exc
    results = payload.get("results") if isinstance(payload, Mapping) else None
    if not isinstance(results, Mapping):
        raise V05LocalError(f"协议基线日志缺少 results: {path}")
    simple_variant = (
        results.get("simple_sync", {}).get("variants", {}).get("main_k6")
    )
    if not isinstance(simple_variant, Mapping):
        raise V05LocalError("协议基线日志缺少 simple_sync/main_k6")
    simple_per_work, simple_macro = _average_work_metrics([simple_variant])
    old_variants = []
    for seed in (42, 43, 44):
        variant = (
            results.get(f"v05_beta_{seed}", {}).get("variants", {}).get("main_k6")
        )
        if isinstance(variant, Mapping):
            old_variants.append(variant)
    if not old_variants:
        raise V05LocalError("协议基线日志缺少 v05_beta_42/43/44/main_k6")
    old_per_work, old_macro = _average_work_metrics(old_variants)
    old_top1 = float(
        np.mean([float(variant["top1"]) for variant in old_variants])
    )
    return {
        "source": str(path),
        "simple_sync": {
            "top1": float(simple_variant["top1"]),
            "macro_top1": simple_macro,
            "per_work": simple_per_work,
        },
        "old_beta": {
            "top1": old_top1,
            "macro_top1": old_macro,
            "per_work": old_per_work,
            "seeds": [42, 43, 44],
        },
    }


def _make_eval_loader(
    records: Any,
    store: FrameDataStore,
    *,
    bin_seconds: float,
    batch_size: int,
    num_workers: int,
    audio_center_shift_seconds: float = 0.0,
    cache_items: bool | None = None,
) -> Any:
    from torch.utils.data import DataLoader

    dataset = LocalPairDataset(
        records,
        store,
        bin_seconds=float(bin_seconds),
        audio_center_shift_seconds=float(audio_center_shift_seconds),
        cache_items=cache_items,
    )
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        collate_fn=collate_local_sets,
    )


def _evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    protocol: Any,
    app_config: Any,
    args: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    model, payload = load_checkpoint(checkpoint_path, device=device)
    model_config = getattr(model, "config", None)
    if model_config is None:
        raise V05LocalError(f"checkpoint 缺少模型配置: {checkpoint_path}")
    store = FrameDataStore(
        visual_root=app_config.paths.data / "visual_dense",
        audio_root=app_config.paths.data / "audio_dense",
        visual_version=args.visual_version,
        audio_version=args.audio_version,
        context_seconds=float(model_config.context_seconds),
        audio_hz=float(model_config.audio_hz),
        bin_seconds=float(protocol.bin_seconds),
        cache_size=max(16, int(args.cache_size)),
    )
    if int(store.visual_dim) != int(model_config.visual_dim):
        raise V05LocalError(
            f"视觉维度不一致: store={store.visual_dim} model={model_config.visual_dim}"
        )
    if int(store.audio_dim) != int(model_config.audio_dim):
        raise V05LocalError(
            f"音频维度不一致: store={store.audio_dim} model={model_config.audio_dim}"
        )
    test_records = list(protocol.split_records()["test"])
    if args.limit is not None:
        test_records = test_records[: int(args.limit)]
    if not test_records:
        raise V05LocalError("完整协议 test split 为空")
    test_loader = _make_eval_loader(
        test_records,
        store,
        bin_seconds=float(protocol.bin_seconds),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        cache_items=True,
    )
    test_metrics = evaluate_local(
        model,
        test_loader,
        device=device,
        k_retrieval=int(args.k_retrieval),
        bootstrap=int(args.bootstrap),
        bootstrap_seed=int(payload.get("seed", 42)) + 1000,
        ablations=DEFAULT_ABLATIONS,
    )
    retrieval_metrics = None
    retrieval_records = list(getattr(protocol, "retrieval_records", ()) or ())
    if args.limit is not None:
        retrieval_records = retrieval_records[: int(args.limit)]
    if retrieval_records:
        retrieval_loader = _make_eval_loader(
            retrieval_records,
            store,
            bin_seconds=float(protocol.bin_seconds),
            batch_size=max(1, int(args.batch_size) // 2),
            num_workers=int(args.num_workers),
            cache_items=False,
        )
        retrieval_metrics = evaluate_local(
            model,
            retrieval_loader,
            device=device,
            k_retrieval=32,
            bootstrap=int(args.bootstrap),
            bootstrap_seed=int(payload.get("seed", 42)) + 2000,
            ablations=("normal",),
        )
    shift_metrics: dict[str, Any] = {}
    for delta in tuple(args.eval_shifts or ()):
        shift = float(delta)
        loader = _make_eval_loader(
            test_records,
            store,
            bin_seconds=float(protocol.bin_seconds),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            audio_center_shift_seconds=shift,
            cache_items=False,
        )
        shift_metrics[f"{shift:+.1f}"] = evaluate_local(
            model,
            loader,
            device=device,
            k_retrieval=int(args.k_retrieval),
            bootstrap=0,
            bootstrap_seed=int(payload.get("seed", 42)) + 3000,
            ablations=("normal",),
        )
    split_works = {
        str(work_id): index
        for index, work_id in enumerate(sorted({int(r.work_id) for r in test_records}))
    }
    return {
        "checkpoint": str(checkpoint_path),
        "arch": str(payload.get("arch", getattr(model, "arch", "unknown"))),
        "seed": int(payload.get("seed", -1)),
        "model_config": model_config.to_dict()
        if hasattr(model_config, "to_dict")
        else {},
        "n_test_anchors": len(test_records),
        "n_test_works": len(split_works),
        "test": test_metrics,
        "retrieval": retrieval_metrics,
        "audio_center_shift": shift_metrics,
    }


def _print_diagnostics(summary: Mapping[str, Any], *, out_path: Path) -> None:
    print("v0.5-beta 局部模型诊断（完整 test；不写 DB/不建 map）")
    for item in summary.get("checkpoints", []):
        print(
            f"  {item.get('arch')} seed={item.get('seed')}: "
            f"top1={item.get('top1')} macro={item.get('macro_top1')} "
            f"content_drop={item.get('audio_content_drop')} "
            f"sync_drop={item.get('audio_sync_drop')} "
            f"pass={item.get('seed_pass')}"
        )
    for arch, bucket in (summary.get("arch_summary") or {}).items():
        print(
            f"  arch={arch}: pass_count={bucket.get('pass_count')} "
            f"arch_pass={bucket.get('arch_pass')}"
        )
    print(f"  overall: {summary.get('conclusion')} (overall_pass={summary.get('overall_pass')})")
    print(f"  JSON: {out_path}")


def _run_evaluation(
    args: argparse.Namespace,
    app_config: Any,
    *,
    device: str,
    protocol_path: Path,
    logs_dir: Path,
) -> int:
    protocol = load_protocol(protocol_path)
    reference = _load_diagnostic_reference(logs_dir)
    checkpoints: list[dict[str, Any]] = []
    for raw_path in args.eval_checkpoints:
        path = Path(str(raw_path)).expanduser()
        if not path.is_file():
            raise V05LocalError(f"checkpoint 不存在: {path}")
        checkpoints.append(
            _evaluate_checkpoint(
                path,
                protocol=protocol,
                app_config=app_config,
                args=args,
                device=device,
            )
        )
    summary = summarize_local_diagnostics(checkpoints, reference=reference)
    out_path = (
        Path(args.eval_out).expanduser()
        if args.eval_out
        else Path(logs_dir) / "v05_local_diagnostics_full.json"
    )
    payload = {
        "created_at": now_utc_iso(),
        "protocol_dir": str(protocol_path),
        "protocol_version": protocol.metadata.get("protocol_version"),
        "split": "test",
        "test_anchor_count": len(protocol.split_records()["test"]),
        "test_work_ids": list(protocol.split_work_ids()["test"]),
        "retrieval_anchor_count": len(protocol.retrieval_records),
        "eval_shifts": [float(value) for value in (args.eval_shifts or ())],
        "reference": reference,
        "criteria": DIAGNOSTIC_CRITERIA,
        "checkpoints": checkpoints,
        "summary": summary,
    }
    _write_json(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        _print_diagnostics(summary, out_path=out_path)
    return 0


def _print_human(summary: Mapping[str, Any], *, out_dir: Path | None = None) -> None:
    print("v0.5-beta 局部帧级时序模型（单一 InfoNCE；不写 DB/不建 map）")
    for arch in summary.get("archs", {}):
        payload = summary["archs"][arch]
        top1 = payload.get("test_top1", {})
        k32 = payload.get("test_top1_k32", {})
        retrieval = payload.get("test_retrieval_top1", {})
        print(
            f"  {arch}: test_top1={top1.get('mean')}±{top1.get('std')} "
            f"(n={top1.get('n')}), top1_k32={k32.get('mean')}±{k32.get('std')}, "
            f"retrieval_k32={retrieval.get('mean')}±{retrieval.get('std')}"
        )
        baseline = payload.get("reference_baselines")
        if baseline:
            print(f"    协议 baselines 引用: {json.dumps(baseline, ensure_ascii=False)}")
    if out_dir is not None:
        print(f"  产物目录: {out_dir}")
    print(f"  JSON: {summary.get('logs')}")


def _smoke(args: argparse.Namespace) -> int:
    import torch

    device = "cpu" if str(args.device) == "auto" else str(args.device)
    torch.manual_seed(0)
    protocol = make_synthetic_protocol(
        n_works=12,
        bins_per_work=40,
        anchors_per_work=3,
        candidates=5,
        seed=int(args.smoke_seed),
    )
    visual_dim = 32
    audio_dim = 24
    batches = build_synthetic_batches(
        protocol=protocol,
        visual_dim=visual_dim,
        audio_dim=audio_dim,
        context_seconds=1.0,
        audio_hz=25.0,
        batch_size=max(2, int(args.batch_size)),
        seed=int(args.smoke_seed),
    )
    model_config = _build_model_config(
        args, visual_dim=visual_dim, audio_dim=audio_dim
    )
    # smoke 用最小 context/1-2 epoch，避免改动真实协议参数
    model_config = LocalModelConfig(
        hidden=model_config.hidden,
        heads=model_config.heads,
        self_layers=model_config.self_layers,
        cross_layers=model_config.cross_layers,
        dropout=0.0,
        context_seconds=1.0,
        audio_hz=25.0,
        visual_dim=visual_dim,
        audio_dim=audio_dim,
        temperature=model_config.temperature,
    )
    out_dir = Path(args.out_dir or (Path.cwd() / "temp" / "v05_local_smoke"))
    out_dir.mkdir(parents=True, exist_ok=True)
    archs = list(ARCHS) if args.arch == "both" else [args.arch]
    summary: dict[str, Any] = {
        "synthetic": True,
        "protocol_records": len(protocol.records),
        "split_counts": {
            name: len(items) for name, items in protocol.split_records().items()
        },
        "archs": {},
        "created_at": now_utc_iso(),
    }
    for arch in archs:
        metrics_list: list[dict[str, Any]] = []
        for seed in [int(value) for value in args.seeds]:
            train_config = _build_train_config(args, model_config, arch)
            train_config = LocalTrainConfig(
                arch=train_config.arch,
                seeds=(seed,),
                epochs=min(2, max(1, int(args.epochs))),
                batch_size=train_config.batch_size,
                lr=train_config.lr,
                weight_decay=train_config.weight_decay,
                patience=0,
                bootstrap=min(20, int(args.bootstrap)),
                k_retrieval=int(args.k_retrieval),
                device=device,
                num_workers=0,
                model=model_config,
            )
            metrics_list.append(
                _run_single(
                    arch=arch,
                    seed=seed,
                    protocol=protocol,
                    loaders=batches,
                    train_config=train_config,
                    device=device,
                    out_dir=out_dir,
                    synthetic=True,
                )
            )
        summary["archs"][arch] = _aggregate(metrics_list, arch)
    logs_path = out_dir / "metrics.json"
    summary["logs"] = str(logs_path)
    _write_json(logs_path, summary)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
    else:
        _print_human(summary, out_dir=out_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/train_v05_local.py",
        description="v0.5-beta 局部帧级时序模型训练（temporal/cross，单一 InfoNCE）。",
    )
    parser.add_argument("--config", default=None, help="配置文件路径（真实数据模式）")
    parser.add_argument("--smoke", action="store_true", help="CPU 合成协议 smoke（不读真实数据）")
    parser.add_argument("--arch", default="both", choices=["both", *ARCHS], help="模型档位")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS), help="训练 seed")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="batch 大小")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP, help="按 work bootstrap 次数")
    parser.add_argument("--k-retrieval", type=int, default=DEFAULT_K_RETRIEVAL, help="K retrieval")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--self-layers", type=int, default=DEFAULT_SELF_LAYERS)
    parser.add_argument("--cross-layers", type=int, default=DEFAULT_CROSS_LAYERS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--context-seconds", type=float, default=DEFAULT_CONTEXT_SECONDS)
    parser.add_argument("--audio-hz", type=float, default=DEFAULT_AUDIO_HZ)
    parser.add_argument("--protocol-dir", default=None, help="协议目录（默认 data/v05_protocol/v2）")
    parser.add_argument("--visual-version", default=None, help="CLIP 8fps 视觉版本目录名")
    parser.add_argument("--audio-version", default=None, help="audio_dense 版本目录名")
    parser.add_argument("--cache-size", type=int, default=8, help="FrameDataStore LRU 作品数")
    parser.add_argument("--limit", type=int, default=None, help="调试：每 split 最多记录数")
    parser.add_argument("--out-dir", default=None, help="产物根目录（默认 data/models）")
    parser.add_argument("--logs-dir", default=None, help="logs 目录（默认 config.paths.logs）")
    parser.add_argument("--smoke-seed", type=int, default=7, help="smoke 合成协议 seed")
    parser.add_argument(
        "--eval-checkpoints",
        nargs="+",
        default=None,
        help="只评估已训练 checkpoint（不训练）；完整协议 test + 修正消融 + 真实错位",
    )
    parser.add_argument(
        "--eval-out",
        default=None,
        help="诊断 JSON 输出（默认 logs/v05_local_diagnostics_full.json）",
    )
    parser.add_argument(
        "--eval-shifts",
        nargs="*",
        type=float,
        default=[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
        help="数据集层面音频相对画面错位秒数（默认 ±0.5/1/2）",
    )
    parser.add_argument("--json", action="store_true", help="只输出 JSON 摘要")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging("INFO")
    if args.smoke:
        try:
            return _smoke(args)
        except (V05LocalError, NebulaError) as exc:
            print(f"局部时序 smoke 失败: {exc}", file=sys.stderr)
            return 1
    try:
        config = load_config(_resolve_config_path(args.config))
    except NebulaError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        print(f"真实训练需要 torch：{exc}", file=sys.stderr)
        return 1
    configure_logging(config.logging.level)
    try:
        device = _resolve_device(str(args.device))
        protocol_path = (
            Path(args.protocol_dir).expanduser()
            if args.protocol_dir
            else config.paths.data / "v05_protocol" / "v2"
        )
        if args.eval_checkpoints:
            logs_dir = (
                Path(args.logs_dir).expanduser()
                if args.logs_dir
                else config.paths.logs
            )
            return _run_evaluation(
                args,
                config,
                device=device,
                protocol_path=protocol_path,
                logs_dir=logs_dir,
            )
        protocol = load_protocol(protocol_path)
        store = FrameDataStore(
            visual_root=config.paths.data / "visual_dense",
            audio_root=config.paths.data / "audio_dense",
            visual_version=args.visual_version,
            audio_version=args.audio_version,
            context_seconds=float(args.context_seconds),
            audio_hz=float(args.audio_hz),
            bin_seconds=float(protocol.bin_seconds),
            cache_size=int(args.cache_size),
        )
        model_config = _build_model_config(
            args, visual_dim=store.visual_dim, audio_dim=store.audio_dim
        )
        archs = list(ARCHS) if args.arch == "both" else [args.arch]
        out_root = (
            Path(args.out_dir).expanduser()
            if args.out_dir
            else config.paths.data / "models"
        )
        logs_dir = (
            Path(args.logs_dir).expanduser()
            if args.logs_dir
            else config.paths.logs
        )
        reference_baselines = _load_reference_baselines(logs_dir)
        summary: dict[str, Any] = {
            "synthetic": False,
            "protocol_dir": str(protocol_path),
            "protocol_version": protocol.metadata.get("protocol_version"),
            "protocol_records": len(protocol.records),
            "protocol_retrieval_records": len(protocol.retrieval_records),
            "protocol_periodic_records": len(protocol.periodic_records),
            "protocol_main_k": protocol.metadata.get("main_k"),
            "protocol_retrieval_k": protocol.metadata.get("retrieval_k"),
            "protocol_offsets": protocol.metadata.get("offsets"),
            "protocol_anchors_counts": protocol.metadata.get("anchors_counts"),
            "protocol_data": protocol.metadata.get("data"),
            "split_counts": {
                name: len(items) for name, items in protocol.split_records().items()
            },
            "split_work_ids": {
                name: list(ids) for name, ids in protocol.split_work_ids().items()
            },
            "reference_baselines": reference_baselines,
            "model_config": model_config.to_dict(),
            "archs": {},
            "created_at": now_utc_iso(),
        }
        for arch in archs:
            metrics_list: list[dict[str, Any]] = []
            for seed in [int(value) for value in args.seeds]:
                train_config = _build_train_config(args, model_config, arch)
                loaders = _split_loaders(
                    protocol,
                    store,
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                    seed=int(seed),
                    limit=args.limit,
                )
                metrics_list.append(
                    _run_single(
                        arch=arch,
                        seed=seed,
                        protocol=protocol,
                        loaders=loaders,
                        train_config=train_config,
                        device=device,
                        out_dir=out_root,
                        synthetic=False,
                        reference_baselines=reference_baselines,
                    )
                )
            summary["archs"][arch] = _aggregate(metrics_list, arch)
            logs_path = logs_dir / f"v05_local_{arch}.json"
            summary_arch = {
                **summary,
                "archs": {arch: summary["archs"][arch]},
                "logs": str(logs_path),
            }
            _write_json(logs_path, summary_arch)
    except (V05LocalError, NebulaError) as exc:
        print(f"局部时序训练失败: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
    else:
        _print_human(summary, out_dir=out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
