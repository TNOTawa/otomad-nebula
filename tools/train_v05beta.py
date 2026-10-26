"""v0.5-beta 局部音画配对 + 双塔对比训练/验证 CLI（研究产物，不注册 DB/地图）。

用法：

    # 真实数据（需先构建 v05_beta_v1 全量 token + t1 audio_dense）
    python tools/train_v05beta.py --dataset-version v05_beta_v1 --audio-version <audio_version> --seeds 3 --device auto

    # CPU 冒烟（合成配对数据 + 小模型，不依赖 token/audio 产物）
    python tools/train_v05beta.py --smoke --epochs 2 --seeds 2 --device cpu

产物：
- data/models/<version>_<seed>/checkpoint.pt
- logs/<version>_<seed>_metrics.json
- logs/<version>_summary.json
- logs/<version>_candidates_manifest.json
不写 DB、不建 map、不写 data/representations。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nebula.config import load_config  # noqa: E402
from nebula.db import Catalog  # noqa: E402
from nebula.errors import NebulaError  # noqa: E402
from nebula.v05_beta import (  # noqa: E402
    BETA_DATASET_VERSION,
    BETA_MODEL_VERSION,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BOOTSTRAP,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN,
    DEFAULT_LR,
    DEFAULT_NEAR_DUP_COS,
    DEFAULT_OFFSETS,
    DEFAULT_PATIENCE,
    DEFAULT_TEMPERATURE,
    DEFAULT_WEIGHT_DECAY,
    BetaConfig,
    V05BetaError,
    run_beta_training,
)

LOGGER = logging.getLogger("train_v05beta")


def _parse_offsets(value: str) -> tuple[int, ...]:
    try:
        offsets = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--offsets 必须是逗号分隔整数，收到 {value!r}") from exc
    if not offsets or any(item < 1 for item in offsets):
        raise argparse.ArgumentTypeError("--offsets 每项必须 >= 1")
    return offsets


def _parse_split_counts(value: str) -> tuple[int, int, int]:
    try:
        items = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--split-counts 必须是 3 个逗号分隔整数，收到 {value!r}"
        ) from exc
    if len(items) != 3 or any(item < 1 for item in items):
        raise argparse.ArgumentTypeError("--split-counts 需要 3 个 >= 1 的整数")
    return items


def _parse_max_segments(value: str) -> int | None:
    text = str(value).strip().lower()
    if text in ("none", "null", "0", "all", "full"):
        return None
    try:
        number = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--max-segments 必须是正整数或 none，收到 {value!r}"
        ) from exc
    if number < 1:
        raise argparse.ArgumentTypeError(f"--max-segments 必须 >= 1，收到 {number}")
    return number


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_v05beta.py",
        description="v0.5-beta 局部音画配对 + 双塔 InfoNCE 训练/验证（研究产物）",
    )
    parser.add_argument("--config", metavar="PATH", default=None, help="配置文件路径（默认当前目录 config.yaml）")
    parser.add_argument("--version", default=BETA_MODEL_VERSION, help="本轮运行/产物版本名（默认 v05_beta）")
    parser.add_argument(
        "--dataset-version",
        default=BETA_DATASET_VERSION,
        help="全量 token 数据集版本（默认 v05_beta_v1，需先构建）",
    )
    parser.add_argument("--audio-version", default=None, help="t1 audio_dense 版本目录名")
    parser.add_argument(
        "--visual-encoder",
        choices=("clip", "dino"),
        default="clip",
        help="视觉 bin 来源（默认 clip；dino 作消融）",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="每 seed 最大 epoch 数")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="每批 anchor 数")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="AdamW 学习率")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="AdamW weight decay")
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN, help="双塔 MLP 隐层维度")
    parser.add_argument("--embed-dim", type=int, default=128, help="双塔输出维度（默认 128）")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="InfoNCE 温度（默认 0.07）")
    parser.add_argument("--seeds", type=int, default=3, help="训练 seed 数（默认 3）")
    parser.add_argument("--seed", type=int, default=42, help="起始 seed（默认 42）")
    parser.add_argument("--split-seed", type=int, default=42, help="划分 hash seed（默认 42）")
    parser.add_argument(
        "--split-counts",
        type=_parse_split_counts,
        default=None,
        metavar="TRAIN,VAL,TEST",
        help="显式指定划分作品数（默认 floor(0.7N)/floor(0.2N)/剩余）",
    )
    parser.add_argument("--offsets", type=_parse_offsets, default=DEFAULT_OFFSETS, help="负样本时间偏移 bins（默认 1,2,4,8,16）")
    parser.add_argument("--near-dup", type=float, default=DEFAULT_NEAR_DUP_COS, help="近重复余弦阈值（默认 0.95）")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP, help="按作品 bootstrap 次数（默认 200）")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="早停 patience（0 关闭）")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int, default=None, help="最多使用多少部 canonical 作品")
    parser.add_argument(
        "--exclude-work-ids",
        type=int,
        nargs="*",
        default=(),
        help="显式排除的作品 ID（例如音频流短于视频、末尾 bin 无效的作品）",
    )
    parser.add_argument(
        "--max-segments",
        type=_parse_max_segments,
        default=None,
        metavar="N|none",
        help="可选 token 截断（默认 none：beta 使用全量不截断）",
    )
    parser.add_argument("--smoke", action="store_true", help="CPU 冒烟：合成配对数据 + 小模型，不依赖真实 token/audio")
    parser.add_argument("--smoke-works", type=int, default=12, help="smoke 合成作品数（默认 12）")
    parser.add_argument("--quiet", action="store_true", help="不打印逐 epoch/逐作品进度")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=None,
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> BetaConfig:
    seeds = tuple(int(args.seed) + index for index in range(max(1, int(args.seeds))))
    return BetaConfig(
        version=args.version,
        dataset_version=args.dataset_version,
        audio_version=args.audio_version,
        visual_encoder=args.visual_encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden=args.hidden,
        embed_dim=args.embed_dim,
        temperature=args.temperature,
        seeds=seeds,
        split_seed=args.split_seed,
        split_counts=args.split_counts,
        offsets=args.offsets,
        near_dup=args.near_dup,
        bootstrap=args.bootstrap,
        patience=args.patience,
        device=args.device,
        limit=args.limit,
        exclude_work_ids=tuple(args.exclude_work_ids or ()),
        max_segments=args.max_segments,
        smoke=bool(args.smoke),
        smoke_works=max(9, int(args.smoke_works)),
    ).validate()


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

        with Catalog(config.paths.db_path) as catalog:
            catalog.initialize()
            result = run_beta_training(config, catalog, cfg, progress=progress)

        summary = result.summary
        aggregate = summary["aggregate"]
        dataset = summary["dataset"]
        print()
        print(
            f"v0.5-beta 数据集: {dataset['dataset_version']} + audio {dataset['audio_version']}"
            f" | 可用作品 {dataset['n_works_available']} | 划分 "
            f"{dataset['split']['counts'][0]}/{dataset['split']['counts'][1]}/{dataset['split']['counts'][2]}"
            f" | visual={dataset['visual_encoder']} | synthetic={dataset['synthetic']}"
        )
        for seed_info in summary["per_seed"]:
            bootstrap = seed_info.get("bootstrap") or {}
            ci = bootstrap.get("ci95") or [None, None]
            print(
                f"  seed={seed_info['seed']}: test_top1={seed_info['test_top1']} "
                f"recall@1={seed_info['recall_at_1']} pos_neg_gap={seed_info['positive_negative_gap']} "
                f"random={seed_info['random_top1']} frozen={seed_info['frozen_projection_top1']} "
                f"simple_sync={seed_info['simple_sync_top1']} "
                f"best_epoch={seed_info['best_epoch']} CI95={ci} "
                f"({seed_info['elapsed_s']:.2f}s)"
            )
        print(
            f"  aggregate: top1 mean={aggregate['test_top1_mean']} std={aggregate['test_top1_std']} "
            f"delta_vs_random={aggregate['delta_vs_random']} "
            f"delta_vs_frozen={aggregate['delta_vs_frozen_projection']} "
            f"delta_vs_simple_sync={aggregate.get('delta_vs_simple_sync')}"
        )
        for limitation in dataset.get("limitations", []):
            print(f"  限制: {limitation}")
        print(f"  summary: {result.summary_path}")
        print(f"  candidates manifest: {result.candidates_manifest_path}")
        for seed_result in result.results:
            print(f"  checkpoint: {seed_result.checkpoint_path}")
            print(f"  metrics:    {seed_result.metrics_path}")
        return 0
    except (V05BetaError, NebulaError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"错误：文件系统操作失败：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("训练已取消", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
