"""v0.4.3 密集视觉语义编码 pilot CLI（不写数据库）。

对一个视频/本地作品，用冻结视觉编码器（CLIP / DINOv2）按指定帧率提取帧级
语义向量，按全片覆盖窗口（默认 8s）池化，保存 dense_visual.npz + manifest，
并打印帧数、解码/编码耗时、实际帧率、存储字节、窗口池化 shape。

用法（项目根目录）::

    .\\.venv\\Scripts\\python.exe tools/dense_visual_pilot.py --local-relpath "某作品.mp4"
    .\\.venv\\Scripts\\python.exe tools/dense_visual_pilot.py --source path/to/v.mp4 ^
        --encoder clip-vit-base-patch16 --fps 8 --limit 5 --out-dir temp/pilot --json

说明：

- --encoder 支持别名（clip-vit-base-patch16 / clip-vit-base-patch32 / dinov2-small
  / dinov2-base），也可以直接给 Hugging Face model id；
- --limit 限制参与窗口池化的窗口数（默认全部），便于快速 pilot；
- 模型下载缓存由配置 model.cache_dir 指定；镜像/环境变量由调用方设置
  （本工具不访问网络策略、不写 DB）；
- 真实模型运行需要 torch/transformers 与权重；默认测试只覆盖 --help 与参数解析。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nebula.config import load_config  # noqa: E402
from nebula.dense_visual import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_FPS,
    DEFAULT_SIZE,
    DenseVisualError,
    HFImageEncoder,
    extract_dense_features,
    pool_windows,
    save_features,
)
from nebula.errors import NebulaError  # noqa: E402
from nebula.log import configure_logging  # noqa: E402
from nebula.sampling import plan_full_coverage  # noqa: E402
from nebula.util import slugify  # noqa: E402

ENCODER_ALIASES: dict[str, str] = {
    "clip-vit-base-patch16": "openai/clip-vit-base-patch16",
    "clip-vit-base-patch32": "openai/clip-vit-base-patch32",
    "clip-vit-large-patch14": "openai/clip-vit-large-patch14",
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
    "dinov2-large": "facebook/dinov2-large",
}
"""常见编码器别名 -> Hugging Face model id。"""


def resolve_encoder_id(value: str) -> str:
    """把编码器别名解析成 Hugging Face model id；未知输入原样返回。"""

    raw = str(value or "").strip()
    if not raw:
        raise DenseVisualError("--encoder 不能为空")
    return ENCODER_ALIASES.get(raw, raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/dense_visual_pilot.py",
        description="密集视觉语义编码 pilot：帧级冻结编码器向量 + 窗口池化（不写 DB）。",
    )
    parser.add_argument("--config", default=None, help="配置文件路径（默认 ./config.yaml）")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source", default=None, help="任意视频文件的绝对/相对路径")
    source_group.add_argument(
        "--local-relpath", default=None, help="input/local 下的相对路径（本地作品）"
    )
    parser.add_argument(
        "--encoder",
        default="clip-vit-base-patch16",
        help="编码器别名或 Hugging Face model id（默认 clip-vit-base-patch16）",
    )
    parser.add_argument("--revision", default=None, help="模型 revision（默认主干）")
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_FPS, help=f"帧级采样率（默认 {DEFAULT_FPS:g}）"
    )
    parser.add_argument(
        "--size", type=int, default=DEFAULT_SIZE, help=f"输入边长（默认 {DEFAULT_SIZE}）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"每批编码帧数（默认 {DEFAULT_BATCH_SIZE}）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多池化多少个窗口（默认全部；仅影响池化与打印，便于快速 pilot）",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="输出目录（默认 temp/dense_visual_pilot/<source 名>）",
    )
    parser.add_argument("--json", action="store_true", help="只输出 JSON 摘要（便于脚本消费）")
    return parser


def _resolve_config_path(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    env_value = os.environ.get("NEBULA_CONFIG")
    if env_value:
        return Path(env_value).expanduser()
    return Path.cwd() / "config.yaml"


def _resolve_source(config, args: argparse.Namespace) -> tuple[Path, str]:
    if args.source:
        path = Path(args.source).expanduser()
        if not path.is_file():
            raise DenseVisualError(f"--source 指向的文件不存在: {path}")
        return path, path.stem
    relpath = str(args.local_relpath or "").strip()
    if not relpath:
        raise DenseVisualError("--local-relpath 不能为空")
    path = config.local_work_path(relpath)
    if not path.is_file():
        raise DenseVisualError(f"本地作品不存在: {path}")
    return path, Path(relpath).stem


def _summary_payload(
    *,
    source: Path,
    encoder: HFImageEncoder,
    features,
    pooled,
    window_count: int,
    limit: int | None,
    saved: dict[str, Path],
    wall_s: float,
) -> dict[str, object]:
    storage_bytes = int(sum(path.stat().st_size for path in saved.values()))
    return {
        "source": str(source),
        "encoder": encoder.model_id,
        "encoder_revision": encoder.revision,
        "features_version": features.version,
        "fps": float(features.fps),
        "size": int(features.size),
        "dim": int(features.dim),
        "n_frames": int(features.n_frames),
        "duration_s": round(float(features.duration_s), 3),
        "decode_s": round(float(features.decode_s), 3),
        "encode_s": round(float(features.encode_s), 3),
        "model_load_s": round(float(features.model_load_s), 3),
        "wall_s": round(float(wall_s), 3),
        "effective_fps": round(float(features.n_frames) / float(features.duration_s), 3),
        "n_windows_total": int(window_count),
        "n_windows_pooled": int(pooled.shape[0]),
        "limit": limit,
        "window_pool_shape": [int(value) for value in pooled.shape],
        "storage_bytes": storage_bytes,
        "files": {name: str(path) for name, path in saved.items()},
    }


def _print_human(payload: dict[str, object]) -> None:
    print("密集视觉语义编码 pilot（无 DB 写入）")
    print(f"  source: {payload['source']}")
    print(
        f"  编码器: {payload['encoder']}"
        f"（revision={payload['encoder_revision']}，version={payload['features_version']}）"
    )
    print(
        f"  帧: {payload['n_frames']} 帧 / {payload['duration_s']}s"
        f"（请求 {payload['fps']}fps，实际 {payload['effective_fps']}fps，dim={payload['dim']}）"
    )
    print(
        f"  耗时: 解码 {payload['decode_s']}s + 编码 {payload['encode_s']}s"
        f" + 模型加载 {payload['model_load_s']}s；墙钟 {payload['wall_s']}s"
    )
    print(
        f"  窗口池化: shape={payload['window_pool_shape']}"
        f"（{payload['n_windows_pooled']}/{payload['n_windows_total']} 个窗口，"
        f"limit={payload['limit']}）"
    )
    print(f"  存储: {payload['storage_bytes']} 字节")
    for name, value in payload["files"].items():
        print(f"    {name}: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(_resolve_config_path(args.config))
    except NebulaError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    configure_logging(config.logging.level)

    encoder: HFImageEncoder | None = None
    try:
        source, source_stem = _resolve_source(config, args)
        model_id = resolve_encoder_id(args.encoder)
        if args.limit is not None and args.limit < 1:
            raise DenseVisualError(f"--limit 必须 >= 1，收到 {args.limit}")
        out_dir = (
            Path(args.out_dir).expanduser()
            if args.out_dir
            else config.paths.temp / "dense_visual_pilot" / slugify(source_stem)
        )
        encoder = HFImageEncoder(
            model_id,
            revision=args.revision,
            cache_dir=config.model.cache_dir,
            device="auto",
            dtype="auto",
        )
        wall_started = time.perf_counter()
        features = extract_dense_features(
            source,
            encoder,
            fps=args.fps,
            size=args.size,
            batch_size=args.batch_size,
            ffmpeg=config.media.ffmpeg,
            timeout_s=config.media.extract_timeout_seconds,
        )
        windows = plan_full_coverage(
            features.duration_s,
            clip_seconds=config.media.clip_seconds,
            overlap_seconds=config.sampling.overlap_seconds,
            min_duration_seconds=config.media.min_duration_seconds,
        )
        window_pairs = [(float(window.start_s), float(window.end_s)) for window in windows]
        if args.limit is not None:
            window_pairs = window_pairs[: args.limit]
        if not window_pairs:
            raise DenseVisualError("没有可池化的窗口（检查 --limit）")
        pooled = pool_windows(features, window_pairs)
        saved = save_features(features, out_dir)
        wall_s = time.perf_counter() - wall_started
    except NebulaError as exc:
        print(f"密集视觉 pilot 失败: {exc}", file=sys.stderr)
        return 1
    finally:
        if encoder is not None:
            encoder.close()

    payload = _summary_payload(
        source=source,
        encoder=encoder,
        features=features,
        pooled=pooled,
        window_count=len(windows),
        limit=args.limit,
        saved=saved,
        wall_s=wall_s,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        _print_human(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
