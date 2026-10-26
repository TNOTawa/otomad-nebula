"""项目统一质量入口（QA gate）。

默认快速离线检查（不访问网络、不加载模型权重）：

1. ruff 静态检查：nebula/、tests/、项目根目录与 tools/ 下的 Python 脚本；
2. pytest：tests/ 全量测试，默认排除 model / network 标记；
3. 文档检查：README.md、ARCHITECTURE.md 存在且包含 v0.2.x / v0.4 / v0.4.3 / v0.4.3.1 / v0.4.4 关键内容；
   docs/adr/ 至少 10 篇（含 0005 身份关联 / 0006 筛选地图 / 0007 高维近邻密度 / 0008 音画错位实验 /
   0009 近邻解释 / 0010 全片序列与表征）并包含
   Decision / Why / Options / Chosen / Consequences / How verified 章节；
4. CLI 入口冒烟：ingest.py、pipeline.py、map.py 及其子命令（map.py variants、
   ingest.py link-local）、tools/clip_stability.py、tools/desync_experiment.py、
   tools/preflight_works.py、tools/build_representation.py、tools/compare_spaces.py、
   tools/events_probe.py、tools/visual_pipeline.py、tools/dense_preprocess_compare.py、
   tools/embedding_diagnostics.py、tools/build_visual_windows.py、tools/build_consensus_graph.py、
   tools/build_v05_tokens.py、tools/train_v05.py、tools/explore_relations.py、
   tools/audio_dense.py、tools/train_v05beta.py 的 --help 均能成功运行；
5. 配置有效性：config.yaml 可加载；可据此生成并加载一份临时隔离配置；ffmpeg/ffprobe 可用；
   v0.4 配置段（sampling/events/temporal 与 representations 目录）与 v0.4.3/v0.4.3.1
   （visual_dense、weight_scheme、visual_window_plan、visual_windows/sources 路径）段有效。

--full 额外运行带 model / network 标记的测试（需要 PE-AV 权重和网络；
       会为子进程设置 NEBULA_MODEL_TESTS=1 / NEBULA_NETWORK_TESTS=1）。

用法::

    .\\.venv\\Scripts\\python.exe qa.py
    .\\.venv\\Scripts\\python.exe qa.py --full
    .\\.venv\\Scripts\\python.exe qa.py --only ruff,pytest
    .\\.venv\\Scripts\\python.exe qa.py --list

退出码：0 = 全部检查通过；1 = 至少一项失败。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
"""qa.py 所在目录，即项目根目录。"""

CHECK_NAMES: tuple[str, ...] = ("ruff", "pytest", "docs", "cli", "config")

CLI_CHECKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ingest.py", ()),
    ("pipeline.py", ()),
    ("map.py", ()),
    ("map.py", ("variants",)),
    ("ingest.py", ("link-local",)),
    ("tools/clip_stability.py", ()),
    ("tools/desync_experiment.py", ()),
    ("tools/preflight_works.py", ()),
    ("tools/build_representation.py", ()),
    ("tools/compare_spaces.py", ()),
    ("tools/events_probe.py", ()),
    ("tools/visual_pipeline.py", ()),
    ("tools/dense_preprocess_compare.py", ()),
    ("tools/embedding_diagnostics.py", ()),
    ("tools/build_visual_windows.py", ()),
    ("tools/build_consensus_graph.py", ()),
    ("tools/build_v05_tokens.py", ()),
    ("tools/train_v05.py", ()),
    ("tools/explore_relations.py", ()),
    ("tools/audio_dense.py", ()),
    ("tools/train_v05beta.py", ()),
)
"""必须能以 --help 成功运行的 CLI 入口（脚本, 子命令参数）。"""

REQUIRED_DOC_FILES: tuple[str, ...] = ("README.md", "ARCHITECTURE.md")

V02_DOC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "README.md": (
        "v0.2",
        "近邻",
        "稳定性",
        "身份关联",
        "筛选",
        "分位",
        "错位",
        "片段级",
        "v0.4",
        "v0.4.3",
        "full_coverage",
        "representations",
        "表征空间",
        "时间轴",
        "visual_dense",
        "密集视觉",
        "letterbox",
        "v0.4.3.1",
        "v0.4.4",
        "visual_windows",
        "window_plan",
        "sources",
        "consensus",
        "coverage",
        "v0.5-alpha",
        "v05_alpha",
        "validation",
        "v05_tokens",
        "v0.4.5",
        "unexpected_relations",
        "v0.5-beta",
        "audio_dense",
    ),
    "ARCHITECTURE.md": (
        "work_links",
        "filter_json",
        "stability",
        "余弦",
        "avsync",
        "分位",
        "坐标不可",
        "representations",
        "sequence_version",
        "full_coverage",
        "representation_id",
        "events",
        "sequence_states",
        "v0.4.3",
        "visual_versions",
        "visual_sequences",
        "visual_dense",
        "dense_visual",
        "source_kind",
        "v0.4.3.1",
        "v0.4.4",
        "visual_window_plans",
        "visual_window_sequences",
        "visual_windows",
        "window_plan",
        "sources",
        "weight_scheme",
        "consensus",
        "v0.5-alpha",
        "v05_alpha",
        "validation",
        "v05_tokens",
        "v0.4.5",
        "unexpected_relations",
        "v0.5-beta",
        "audio_dense",
    ),
}
"""v0.2.x / v0.4 文档必须覆盖的关键内容（存在性检查，不校验具体措辞）。"""

REQUIRED_ADR_PREFIXES: tuple[str, ...] = ("0005", "0006", "0007", "0008", "0009", "0010")
"""必需的 ADR 编号前缀（身份关联、筛选地图、高维近邻密度、错位实验、近邻解释、全片表征）。"""

MIN_ADR_COUNT = 10
ADR_SECTIONS: tuple[str, ...] = (
    "Decision",
    "Why",
    "Options",
    "Chosen",
    "Consequences",
    "How verified",
)

COMMAND_TIMEOUT_S = 1800.0
CLI_TIMEOUT_S = 180.0


@dataclass
class CheckResult:
    """一项检查的结果。"""

    name: str
    ok: bool
    detail: str
    duration_s: float = 0.0
    output: str = field(default="", repr=False)


# ──────────────────────────────────────────────────────────────────────────────
# 基础工具
# ──────────────────────────────────────────────────────────────────────────────


def project_python() -> Path:
    """优先使用项目 .venv 里的 Python，找不到时退回当前解释器。"""

    candidates = (
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT / ".venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return Path(sys.executable)


def find_ruff() -> Path | None:
    """优先使用与项目 Python 同目录的 ruff，其次 PATH 中的 ruff。"""

    env_candidate = project_python().parent / "ruff.exe"
    if env_candidate.is_file():
        return env_candidate
    env_candidate = project_python().parent / "ruff"
    if env_candidate.is_file():
        return env_candidate
    found = shutil.which("ruff")
    return Path(found) if found else None


def _child_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    if extra:
        env.update(extra)
    return env


def run_command(
    command: list[str | Path],
    *,
    timeout_s: float,
    cwd: Path = ROOT,
    extra_env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """运行子进程并捕获文本输出（统一 UTF-8，避免 Windows 控制台乱码）。"""

    return subprocess.run(
        [str(part) for part in command],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
        env=_child_env(extra_env),
        check=False,
    )


def _tail(text: str, max_lines: int = 40, max_chars: int = 6000) -> str:
    lines = (text or "").strip().splitlines()
    tail = "\n".join(lines[-max_lines:])
    if len(tail) > max_chars:
        tail = "..." + tail[-max_chars:]
    return tail


def _last_meaningful_line(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


# ──────────────────────────────────────────────────────────────────────────────
# 具体检查
# ──────────────────────────────────────────────────────────────────────────────


def check_ruff(*, verbose: bool = False) -> CheckResult:
    started = time.perf_counter()
    ruff = find_ruff()
    if ruff is None:
        return CheckResult(
            "ruff",
            False,
            "未找到 ruff；请确认 .venv 已安装 dev 依赖，或用 .venv 的 Python 运行 qa.py",
            time.perf_counter() - started,
        )
    top_level = sorted(path.name for path in ROOT.glob("*.py") if path.is_file())
    targets = ["nebula", "tests", *top_level]
    command = [ruff, "check", *targets]
    try:
        proc = run_command(command, timeout_s=COMMAND_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return CheckResult(
            "ruff",
            False,
            f"ruff 检查超时（>{COMMAND_TIMEOUT_S:.0f}s）",
            time.perf_counter() - started,
        )
    duration = time.perf_counter() - started
    output = (proc.stdout or "") + (proc.stderr or "")
    if verbose:
        print(f"  命令: {' '.join(map(str, command))}")
    if proc.returncode == 0:
        summary = _last_meaningful_line(proc.stdout) or "无问题"
        targets_text = "nebula, tests, " + (f"{len(top_level)} 个顶层脚本" if top_level else "无顶层脚本")
        detail = f"{summary}（{targets_text}）"
        return CheckResult("ruff", True, detail, duration, output)
    return CheckResult(
        "ruff",
        False,
        f"ruff 返回 {proc.returncode}；{_last_meaningful_line(output) or '见下方输出'}",
        duration,
        output,
    )


def check_pytest(*, full: bool = False, verbose: bool = False) -> CheckResult:
    started = time.perf_counter()
    python = project_python()
    command: list[str | Path] = [python, "-m", "pytest", "tests"]
    extra_env: dict[str, str] | None = None
    mode = "全部标记（model/network 已启用）"
    if not full:
        command += ["-m", "not model and not network"]
        mode = "排除 model/network"
    else:
        # 默认离线测试用 skipif 守卫；完整模式需要显式打开这两个环境开关，
        # 否则会“收集到但全部 skip”，门禁看似通过却未真正验证模型/网络。
        extra_env = {"NEBULA_MODEL_TESTS": "1", "NEBULA_NETWORK_TESTS": "1"}
    try:
        proc = run_command(command, timeout_s=COMMAND_TIMEOUT_S, extra_env=extra_env)
    except subprocess.TimeoutExpired:
        return CheckResult(
            "pytest",
            False,
            f"pytest 超时（>{COMMAND_TIMEOUT_S:.0f}s）",
            time.perf_counter() - started,
        )
    duration = time.perf_counter() - started
    output = (proc.stdout or "") + (proc.stderr or "")
    if verbose:
        print(f"  命令: {' '.join(map(str, command))}")
    if proc.returncode == 0:
        summary = _last_meaningful_line(proc.stdout) or "全部通过"
        return CheckResult("pytest", True, f"{summary}（{mode}）", duration, output)
    if proc.returncode == 5:
        return CheckResult("pytest", False, "没有收集到任何测试（pytest exit 5）", duration, output)
    summary = _last_meaningful_line(output)
    detail = f"pytest 返回 {proc.returncode}"
    if summary:
        detail += f"；{summary}"
    return CheckResult("pytest", False, detail, duration, output)


def check_docs(*, verbose: bool = False) -> CheckResult:
    started = time.perf_counter()
    problems: list[str] = []
    for name in REQUIRED_DOC_FILES:
        path = ROOT / name
        if not path.is_file():
            problems.append(f"缺少 {name}")
        elif path.stat().st_size < 100:
            problems.append(f"{name} 内容过少（{path.stat().st_size} 字节）")

    for name, keywords in V02_DOC_KEYWORDS.items():
        path = ROOT / name
        if not path.is_file():
            continue  # 缺失或过短已由上面的文档文件检查报告
        try:
            lowered = path.read_text(encoding="utf-8").lower()
        except OSError as exc:
            problems.append(f"无法读取 {name}: {exc}")
            continue
        missing = [keyword for keyword in keywords if keyword.lower() not in lowered]
        if missing:
            problems.append(f"{name} 缺少 v0.2.x/v0.4 内容: {', '.join(missing)}")

    adr_dir = ROOT / "docs" / "adr"
    adr_files = sorted(path for path in adr_dir.glob("*.md") if path.name[:1].isdigit())
    if len(adr_files) < MIN_ADR_COUNT:
        problems.append(f"docs/adr/ 至少需要 {MIN_ADR_COUNT} 篇 ADR，当前 {len(adr_files)} 篇")
    adr_names = [path.name for path in adr_files]
    for prefix in REQUIRED_ADR_PREFIXES:
        if not any(name.startswith(prefix) for name in adr_names):
            problems.append(f"docs/adr/ 缺少 {prefix} 开头的 ADR（必需决策）")
    for adr in adr_files:
        try:
            text = adr.read_text(encoding="utf-8")
        except OSError as exc:
            problems.append(f"无法读取 {adr.name}: {exc}")
            continue
        missing = [section for section in ADR_SECTIONS if section.lower() not in text.lower()]
        if missing:
            problems.append(f"{adr.name} 缺少章节: {', '.join(missing)}")

    duration = time.perf_counter() - started
    if verbose:
        print(f"  ADR 文件: {[path.name for path in adr_files]}")
    if problems:
        return CheckResult("docs", False, "；".join(problems), duration)
    detail = (
        f"README/ARCHITECTURE 含 v0.2.x/v0.4/v0.4.3/v0.4.3.1/v0.4.4/v0.5-alpha/"
        f"v0.4.5/v0.5-beta 内容，ADR {len(adr_files)} 篇且章节完整"
    )
    return CheckResult("docs", True, detail, duration)


def check_cli(*, verbose: bool = False) -> CheckResult:
    started = time.perf_counter()
    python = project_python()
    problems: list[str] = []
    passed: list[str] = []
    outputs: list[str] = []
    for script_name, sub_args in CLI_CHECKS:
        script = ROOT / script_name
        label = " ".join((script_name, *sub_args))
        if not script.is_file():
            problems.append(f"缺少入口脚本 {script_name}")
            continue
        command = [python, script, *sub_args, "--help"]
        try:
            proc = run_command(command, timeout_s=CLI_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            problems.append(f"{label} --help 超时（>{CLI_TIMEOUT_S:.0f}s）")
            continue
        combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        if verbose:
            print(f"  命令: {' '.join(map(str, command))}")
        if proc.returncode != 0:
            problems.append(f"{label} --help 退出码 {proc.returncode}")
            outputs.append(f"--- {label} --help ---\n{_tail(combined)}")
        elif "usage" not in combined.lower():
            problems.append(f"{label} --help 输出中没有 usage 信息")
            outputs.append(f"--- {label} --help ---\n{_tail(combined)}")
        else:
            passed.append(label)
    duration = time.perf_counter() - started
    if problems:
        return CheckResult("cli", False, "；".join(problems), duration, "\n".join(outputs))
    detail = (
        f"{len(passed)} 个 CLI 入口 --help 均可运行"
        "（含 map.py variants、ingest.py link-local、tools/clip_stability.py、"
        "tools/desync_experiment.py、tools/preflight_works.py、"
        "tools/build_representation.py、tools/compare_spaces.py、tools/events_probe.py、"
        "tools/visual_pipeline.py、tools/dense_preprocess_compare.py、"
        "tools/embedding_diagnostics.py、tools/build_visual_windows.py、"
        "tools/build_consensus_graph.py、tools/build_v05_tokens.py、tools/train_v05.py、"
        "tools/explore_relations.py、tools/audio_dense.py、tools/train_v05beta.py）"
    )
    return CheckResult("cli", True, detail, duration)


def check_config(*, verbose: bool = False) -> CheckResult:
    started = time.perf_counter()
    try:
        from nebula.config import load_config

        config_path = ROOT / "config.yaml"
        cfg = load_config(config_path)
    except Exception as exc:  # noqa: BLE001 - QA 需要把任何加载失败变成可读结果
        detail = f"config.yaml 无法加载: {type(exc).__name__}: {exc}"
        return CheckResult("config", False, detail, time.perf_counter() - started)

    problems: list[str] = []
    if not cfg.embedding_version:
        problems.append("embedding_version 为空")
    if cfg.sampling.mode not in ("legacy", "full_coverage"):
        problems.append(f"sampling.mode 非法: {cfg.sampling.mode!r}")
    if not cfg.events.version:
        problems.append("events.version 为空")
    if not 2 <= cfg.temporal.stages <= 8:
        problems.append(f"temporal.stages 超出 2..8: {cfg.temporal.stages}")
    if cfg.sampling.mode == "full_coverage" and cfg.sampling.overlap_seconds >= cfg.media.clip_seconds:
        problems.append("full_coverage 的 overlap_seconds 必须小于 clip_seconds")
    if not cfg.paths.representations.is_dir():
        problems.append(f"representations 目录不存在: {cfg.paths.representations}")
    visual = cfg.visual_dense
    if not 0 < visual.fps <= 120:
        problems.append(f"visual_dense.fps 非法: {visual.fps}")
    if visual.size < 32 or visual.size % 2 != 0:
        problems.append(f"visual_dense.size 必须是 >=32 的偶数: {visual.size}")
    if visual.preprocess not in ("stretch", "letterbox"):
        problems.append(f"visual_dense.preprocess 非法: {visual.preprocess!r}")
    if visual.batch_size < 1:
        problems.append(f"visual_dense.batch_size 非法: {visual.batch_size}")
    if visual.subset_fps <= visual.fps:
        problems.append(
            f"visual_dense.subset_fps({visual.subset_fps}) 必须大于 fps({visual.fps})"
        )
    if not visual.encoders:
        problems.append("visual_dense.encoders 不能为空")
    for position, encoder in enumerate(visual.encoders):
        if not encoder.model.strip():
            problems.append(f"visual_dense.encoders[{position}].model 为空")
        if not encoder.revision.strip():
            problems.append(f"visual_dense.encoders[{position}].revision 为空（必须固定 revision）")
    if not cfg.paths.visual_dense.is_dir():
        problems.append(f"visual_dense 目录不存在: {cfg.paths.visual_dense}")
    if visual.weight_scheme not in ("coverage-v1", "equal"):
        problems.append(f"visual_dense.weight_scheme 非法: {visual.weight_scheme!r}")
    window_plan = cfg.visual_window_plan_version
    if not window_plan:
        problems.append("visual_window_plan_version 为空")
    elif visual.weight_scheme not in window_plan:
        problems.append(
            f"visual_window_plan_version 未包含权重方案 {visual.weight_scheme!r}: {window_plan!r}"
        )
    if not str(cfg.paths.visual_dense).startswith(str(cfg.paths.data)):
        problems.append("visual_dense 目录不在 data 目录内")
    for label, path in (
        ("visual_windows", cfg.paths.visual_windows),
        ("sources", cfg.paths.sources),
    ):
        if not str(path).startswith(str(cfg.paths.data)):
            problems.append(f"{label} 目录不在 data 目录内: {path}")
        if str(path).startswith(str(cfg.paths.temp)):
            problems.append(f"{label} 目录不能位于 temp 下: {path}")
    if not cfg.paths.visual_windows.is_dir():
        problems.append(f"visual_windows 目录不存在: {cfg.paths.visual_windows}")
    if not cfg.paths.sources.is_dir():
        problems.append(f"sources 目录不存在: {cfg.paths.sources}")
    if cfg.legacy_embedding_version == cfg.full_sequence_version:
        problems.append("legacy 与 full 的序列版本号不应相同")
    if not str(cfg.paths.db_path).startswith(str(cfg.paths.data)):
        problems.append("db_path 不在 data 目录内")
    for tool_name in (cfg.media.ffmpeg, cfg.media.ffprobe):
        if shutil.which(tool_name) is None:
            problems.append(f"PATH 中找不到 {tool_name}")

    temp_detail = ""
    try:
        from tests.fixtures.video_factory import create_temp_config

        with tempfile.TemporaryDirectory(prefix="nebula-qa-config-") as temp_dir:
            tmp_root = Path(temp_dir)
            temp_cfg = create_temp_config(tmp_root)
            expected_data = (tmp_root / "data").resolve()
            if temp_cfg.paths.data != expected_data:
                problems.append("临时配置的 data 路径没有落在临时目录内")
            if not temp_cfg.paths.input_local.is_dir():
                problems.append("临时配置未创建 input/local 目录")
            temp_detail = "；临时配置可生成且路径已隔离到临时目录"
    except Exception as exc:  # noqa: BLE001 - 同上
        problems.append(f"临时配置生成失败: {type(exc).__name__}: {exc}")

    duration = time.perf_counter() - started
    if verbose:
        print(f"  embedding_version = {cfg.embedding_version}")
    if problems:
        return CheckResult("config", False, "；".join(problems), duration)
    detail = f"config.yaml 有效（embedding_version={cfg.embedding_version}）{temp_detail}"
    return CheckResult("config", True, detail, duration)


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="音MAD星云计划统一质量入口：ruff + pytest + 文档/入口/配置检查。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--full", action="store_true", help="额外运行 model/network 标记测试")
    parser.add_argument(
        "--only",
        default="",
        help="只运行指定检查（逗号分隔）: " + ",".join(CHECK_NAMES),
    )
    parser.add_argument("--list", action="store_true", help="列出全部检查项后退出")
    parser.add_argument("--verbose", action="store_true", help="打印执行的命令与更多输出")
    return parser.parse_args(argv)


def _selected_checks(only: str) -> list[str]:
    if not only.strip():
        return list(CHECK_NAMES)
    selected = [part.strip() for part in only.split(",") if part.strip()]
    unknown = [name for name in selected if name not in CHECK_NAMES]
    if unknown:
        raise SystemExit(
            f"--only 含未知检查项: {', '.join(unknown)}；可选: {', '.join(CHECK_NAMES)}"
        )
    return selected


def _print_result(result: CheckResult) -> None:
    flag = "PASS" if result.ok else "FAIL"
    print(f"[{flag}] {result.name:<7} {result.duration_s:6.2f}s  {result.detail}")


def _force_utf8_output() -> None:
    """把标准输出/错误切成 UTF-8，避免 Windows 控制台把中文打成乱码。"""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - 非常规流对象
                continue


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()
    args = _parse_args(argv)
    if args.list:
        print("\n".join(CHECK_NAMES))
        return 0
    selected = _selected_checks(args.only)
    mode = "完整模式（包含 model/network）" if args.full else "快速离线模式"
    print(f"音MAD星云计划 QA · {mode}")
    print(f"项目根目录: {ROOT}")
    print(f"Python: {project_python()}")
    print("-" * 72)

    runners = {
        "ruff": lambda: check_ruff(verbose=args.verbose),
        "pytest": lambda: check_pytest(full=args.full, verbose=args.verbose),
        "docs": lambda: check_docs(verbose=args.verbose),
        "cli": lambda: check_cli(verbose=args.verbose),
        "config": lambda: check_config(verbose=args.verbose),
    }

    results: list[CheckResult] = []
    total_started = time.perf_counter()
    for name in selected:
        try:
            result = runners[name]()
        except Exception as exc:  # noqa: BLE001 - 单项崩溃不应中断整个质量入口
            result = CheckResult(name, False, f"检查本身崩溃: {type(exc).__name__}: {exc}")
            if args.verbose:
                traceback.print_exc()
        results.append(result)
        _print_result(result)
        if not result.ok and result.output:
            print("  " + "\n  ".join(_tail(result.output).splitlines()))

    total_duration = time.perf_counter() - total_started
    failed = [result for result in results if not result.ok]
    print("-" * 72)
    print(
        f"汇总: {len(results) - len(failed)} 通过 / {len(failed)} 失败"
        f" / 总耗时 {total_duration:.2f}s"
    )
    if failed:
        print("失败项: " + ", ".join(result.name for result in failed))
        print("请先修复失败项再提交；完整模式请运行: python qa.py --full")
        return 1
    print("全部检查通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
