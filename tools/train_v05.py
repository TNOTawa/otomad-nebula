"""v0.5-alpha 分层音画时序模型的训练/验证 CLI（研究产物，不注册 DB、不建地图）。

用法：

    python tools/train_v05.py --version v05_alpha_v1 --epochs 30 --device auto
    python tools/train_v05.py --smoke --epochs 1 --batch-size 2 --limit 6 --device cpu

依赖：
- 数据层（task-33）：data/v05_tokens/<dataset_version>/<work_id>.npz；未就绪时 --smoke
  会使用合成 tokens 跑通训练/验证链路。
- 模型层（task-34）：nebula.v05_model.V05AlphaModel/V05AlphaConfig；未就绪时只能
  --smoke（使用本模块内的小模型）。

产物：
- data/models/<version>/checkpoint.pt
- logs/<version>_metrics.json
- data/representations/transformer_alpha_v1/<sequence_version>/vectors.npy（研究产物，
  不写 representations 表、不建 map）
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
from nebula.v05_train import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASET_VERSION,
    DEFAULT_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_VERSION,
    TrainConfig,
    V05TrainError,
    run_training,
)

LOGGER = logging.getLogger("train_v05")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_v05.py",
        description="v0.5-alpha 分层音画时序模型训练/验证（研究产物，不注册 DB）",
    )
    parser.add_argument("--config", metavar="PATH", default=None, help="配置文件路径（默认当前目录 config.yaml）")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="模型版本/产物目录名（默认 v05_alpha_v1）")
    parser.add_argument(
        "--dataset-version",
        default=DEFAULT_DATASET_VERSION,
        help="data/v05_tokens 下的数据集版本（默认 v05_alpha_v1）",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="最大 epoch 数（默认 30）")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="每批作品数（默认 4）")
    parser.add_argument("--lr", type=float, default=None, help="AdamW 学习率（默认 1e-3）")
    parser.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay（默认 0.01）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="训练设备（默认 auto）")
    parser.add_argument("--limit", type=int, default=None, help="最多使用多少部作品（按 work_id 升序）")
    parser.add_argument("--mask-ratio", type=float, default=None, help="masked segment 比例（默认 0.2）")
    parser.add_argument("--max-segments", type=int, default=None, help="每部作品最多段数（默认 128）")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="早停 patience（0 关闭，默认 5）")
    parser.add_argument("--hidden", type=int, default=None, help="模型 hidden 维覆盖（smoke 默认 32）")
    parser.add_argument("--local-layers", type=int, default=None, help="局部 Transformer 层数覆盖")
    parser.add_argument("--temporal-layers", type=int, default=None, help="时序 Transformer 层数覆盖")
    parser.add_argument("--dropout", type=float, default=None, help="dropout 覆盖")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="CPU 快速冒烟：合成 tokens + 微型合成模型，不依赖 task-33/34 产物",
    )
    parser.add_argument("--quiet", action="store_true", help="不打印逐 epoch 进度")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=None,
        help="日志级别（默认取 config.logging.level）",
    )
    return parser


def _train_config(args: argparse.Namespace) -> TrainConfig:
    defaults = TrainConfig()
    return TrainConfig(
        version=args.version,
        dataset_version=args.dataset_version,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=defaults.lr if args.lr is None else args.lr,
        weight_decay=(
            defaults.weight_decay if args.weight_decay is None else args.weight_decay
        ),
        seed=args.seed,
        mask_ratio=defaults.mask_ratio if args.mask_ratio is None else args.mask_ratio,
        max_segments=(
            defaults.max_segments if args.max_segments is None else args.max_segments
        ),
        patience=args.patience,
        device=args.device,
        limit=args.limit,
        smoke=bool(args.smoke),
        hidden=args.hidden,
        local_layers=args.local_layers,
        temporal_layers=args.temporal_layers,
        dropout=args.dropout,
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
        cfg = _train_config(args)

        def progress(text: str) -> None:
            if not args.quiet:
                print(text, flush=True)

        with Catalog(config.paths.db_path) as catalog:
            catalog.initialize()
            result = run_training(config, catalog, cfg, progress=progress)

        metrics = result.metrics
        val = metrics["val"]
        training = metrics["training"]
        print()
        print(result.summary_text)
        print(
            f"  val: mask_cos={val['mask_cosine']} mask_mse={val['mask_mse']} "
            f"full_mean_baseline={val['full_mean_baseline_cosine']} "
            f"av_acc={val['av_accuracy']} av_auc={val['av_auc']} "
            f"order_acc={val['order_accuracy']}"
        )
        sensitivity = metrics["temporal_sensitivity"]
        print(
            f"  时间敏感性: normal={sensitivity['normal_mask_cosine']} "
            f"reversed={sensitivity['reversed_mask_cosine']} "
            f"shuffled={sensitivity['shuffled_mask_cosine']} "
            f"reverse_drop={sensitivity['reverse_drop']} "
            f"shuffle_drop={sensitivity['shuffle_drop']}"
        )
        representation = metrics["representation"]
        print(
            f"  表征: n={representation['n_works']} "
            f"pairwise_mean={representation['pairwise_cosine_mean']} "
            f"effective_rank={representation['effective_rank']} "
            f"top5_jaccard_full_mean={representation['top5_jaccard_full_mean']} "
            f"duration_r2={representation['duration_r2']} "
            f"author_bal_acc={representation['author_balanced_accuracy']}"
        )
        print(
            f"  最佳 epoch={training['best_epoch']} "
            f"early_stopped={training['early_stopped']}"
        )
        print(f"  checkpoint: {result.checkpoint_path}")
        print(f"  metrics:    {result.metrics_path}")
        print(f"  vectors:    {result.vectors_path}")
        return 0
    except (V05TrainError, NebulaError) as exc:
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
