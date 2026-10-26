"""v0.5-beta 协议 v2 构建 + 旧 checkpoint 重评 CLI。

流程：
1. 读取 beta 官方 split（74/21/12）、全量 token（v05_beta_v1）与 audio_dense；
2. 构建固定 K=6 主协议（1 正 + 5 双向偏移负样本）、periodic 独立集、test K=32 retrieval；
3. 协议 manifest/anchor 记录写到 data/v05_protocol/v2；
4. 在相同协议上重评旧 checkpoints v05_beta_42/43/44 + random/frozen/simple_sync；
5. 输出 logs/v05_protocol_v2_metrics.json + CSV。

不训练新模型、不写 DB/不建 map/不写 representations。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nebula.config import load_config  # noqa: E402
from nebula.errors import NebulaError  # noqa: E402
from nebula.v05_protocol import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_BOOTSTRAP,
    MAIN_K,
    NEAR_DUP_COSINE,
    PROTOCOL_VERSION,
    RETRIEVAL_K,
    SPLIT_SEED,
    TOKENS_VERSION,
    ProtocolConfig,
    V05ProtocolError,
    build_protocol,
    evaluate_protocol,
    save_protocol,
)

LOGGER = logging.getLogger("build_v05_protocol")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_v05_protocol.py",
        description="v0.5-beta 协议 v2：固定 K 双向候选 + 旧 checkpoint 重评（不训练）",
    )
    parser.add_argument("--config", metavar="PATH", default=None, help="配置文件路径（默认当前目录 config.yaml）")
    parser.add_argument("--version", default=PROTOCOL_VERSION, help="协议版本（默认 v2）")
    parser.add_argument("--tokens-version", default=TOKENS_VERSION, help="全量 token 版本（默认 v05_beta_v1）")
    parser.add_argument("--audio-version", default=None, help="audio_dense 版本（默认取 beta summary 中的版本）")
    parser.add_argument(
        "--visual-encoder",
        choices=("clip", "dino"),
        default="clip",
        help="视觉特征来源（默认 clip，与旧 checkpoints 一致）",
    )
    parser.add_argument("--seed", type=int, default=SPLIT_SEED, help="候选采样/协议种子（默认 42）")
    parser.add_argument("--near-dup", type=float, default=NEAR_DUP_COSINE, help="近重复余弦阈值（默认 0.95）")
    parser.add_argument("--main-k", type=int, default=MAIN_K, help="主协议固定候选数 K（默认 6）")
    parser.add_argument("--retrieval-k", type=int, default=RETRIEVAL_K, help="retrieval 候选池大小（默认 32）")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="评估 batch anchor 数")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP, help="按作品 bootstrap 次数")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        metavar="PATH",
        help="旧 checkpoint 列表（默认 data/models/v05_beta_{42,43,44}/checkpoint.pt）",
    )
    parser.add_argument("--build-only", action="store_true", help="只构建协议 manifest/anchor 记录，不评估")
    parser.add_argument("--quiet", action="store_true", help="不打印逐作品构建进度")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=None,
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> ProtocolConfig:
    checkpoints = tuple(str(path) for path in (args.checkpoints or ()))
    return ProtocolConfig(
        version=args.version,
        tokens_version=args.tokens_version,
        audio_version=args.audio_version,
        visual_encoder=args.visual_encoder,
        seed=args.seed,
        near_dup=args.near_dup,
        main_k=args.main_k,
        retrieval_k=args.retrieval_k,
        batch_size=args.batch_size,
        device=args.device,
        bootstrap=args.bootstrap,
        checkpoint_paths=checkpoints,
    ).validate()


def _format_value(value: Any, *, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _print_protocol_summary(manifest: Mapping[str, Any], *, manifest_path: Path) -> None:
    split = manifest.get("split", {})
    counts = [int(split.get(name, {}).get("count", 0)) for name in ("train", "val", "test")]
    distribution = manifest.get("candidate_distribution", {})
    main = distribution.get("main", {})
    periodic = distribution.get("periodic", {})
    retrieval = distribution.get("retrieval", {})
    print()
    print("v0.5-beta 协议 v2")
    print(f"  manifest: {manifest_path}")
    print(
        f"  split: {counts[0]}/{counts[1]}/{counts[2]}（beta 固定划分）"
        f" | 评估 split: {manifest.get('evaluation_split')}"
    )
    print(
        f"  主协议: K={main.get('k')} kept={main.get('kept')}/{main.get('anchors_total')}"
        f" | 候选数分布: {main.get('candidate_count_hist')}"
        f" | 可用负数分布: {main.get('available_negatives_hist', {})}"
    )
    print(
        f"  periodic: anchors={periodic.get('anchors')}"
        f" | chance_mean={periodic.get('chance_top1_mean')}"
        f" | 候选数分布: {periodic.get('candidate_count_hist')}"
    )
    print(
        f"  retrieval: K={retrieval.get('k')} anchors={retrieval.get('anchors')}"
        f" | chance_top1={retrieval.get('chance_top1')}"
    )


def _print_metrics(metrics: Mapping[str, Any]) -> None:
    rows = metrics.get("rows") or []
    print()
    print("旧 checkpoint + 基线（同一协议 v2）")
    header = (
        f"{'model':<22} {'variant':<15} {'K':>3} {'chance':>8} {'top1':>8} "
        f"{'R@1':>8} {'R@5':>8} {'R@10':>8} {'n':>7} {'ci95':>20}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        ci = (
            f"[{_format_value(row.get('ci95_low'))}, {_format_value(row.get('ci95_high'))}]"
            if row.get("ci95_low") is not None
            else ""
        )
        print(
            f"{str(row.get('model')):<22} {str(row.get('variant')):<15} "
            f"{int(row.get('k') or 0):>3} {_format_value(row.get('chance_top1')):>8} "
            f"{_format_value(row.get('top1')):>8} {_format_value(row.get('recall_at_1')):>8} "
            f"{_format_value(row.get('recall_at_5')):>8} {_format_value(row.get('recall_at_10')):>8} "
            f"{row.get('n_anchors') or 0:>7} {ci:>20}"
        )
    print()
    print("说明：")
    for note in metrics.get("notes", []):
        print(f"  - {note}")
    artifacts = metrics.get("artifacts", {})
    print(f"  metrics: {artifacts.get('metrics')}")
    print(f"  csv:     {artifacts.get('csv')}")
    print(f"  manifest:{artifacts.get('protocol_manifest')}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        config = load_config(args.config) if args.config else load_config()
        logging.basicConfig(
            level=getattr(
                logging,
                str(args.log_level or config.logging.level).upper(),
                logging.INFO,
            ),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        config.ensure_directories()
        cfg = _config_from_args(args)

        def progress(text: str) -> None:
            if not args.quiet:
                print(text, flush=True)

        bundle = build_protocol(config, cfg, progress=progress)
        if args.build_only:
            saved = save_protocol(bundle, config)
            print()
            print(f"协议已写入: {saved['manifest_path']}")
            print(f"anchor 记录: {saved['anchors_path']}（sha256={saved['anchors_sha256'][:16]}…）")
            return 0
        metrics = evaluate_protocol(config, bundle, cfg, progress=progress)
        _print_protocol_summary(bundle.manifest, manifest_path=Path(metrics["protocol_manifest"]))
        _print_metrics(metrics)
        return 0
    except (V05ProtocolError, NebulaError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except (OSError, ImportError) as exc:
        print(f"错误：运行环境/文件系统失败：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("已取消", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
