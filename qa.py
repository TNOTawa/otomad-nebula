"""项目统一质量入口（QA gate）。

默认快速离线检查（不访问网络、不加载模型权重）：

1. ruff 静态检查：nebula/、tests/、项目根目录全部 Python 脚本；
2. pytest：tests/ 全量测试，默认排除 model / network 标记；
3. 文档检查：README.md、ARCHITECTURE.md 存在且非空，docs/adr/ 至少 3 篇
   并包含 Decision / Why / Options / Chosen / Consequences / How verified 章节；
4. CLI 入口冒烟：ingest.py、pipeline.py、map.py 的 --help 均能成功运行；
5. 配置有效性：config.yaml 可加载；可据此生成并加载一份临时隔离配置；ffmpeg/ffprobe 可用。

--full 额外运行带 model / network 标记的测试（需要 PE-AV 权重和网络）。

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
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
"""qa.py 所在目录，即项目根目录。"""

CHECK_NAMES: tuple[str, ...] = ("ruff", "pytest", "docs", "cli", "config")

ENTRY_SCRIPTS: tuple[str, ...] = ("ingest.py", "pipeline.py", "map.py")
"""必须提供可运行 --help 的顶层 CLI 脚本。"""

REQUIRED_DOC_FILES: tuple[str, ...] = ("README.md", "ARCHITECTURE.md")

MIN_ADR_COUNT = 3
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


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    return env


def run_command(
    command: list[str | Path],
    *,
    timeout_s: float,
    cwd: Path = ROOT,
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
        env=_child_env(),
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
    mode = "全部标记"
    if not full:
        command += ["-m", "not model and not network"]
        mode = "排除 model/network"
    try:
        proc = run_command(command, timeout_s=COMMAND_TIMEOUT_S)
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

    adr_dir = ROOT / "docs" / "adr"
    adr_files = sorted(path for path in adr_dir.glob("*.md") if path.name[:1].isdigit())
    if len(adr_files) < MIN_ADR_COUNT:
        problems.append(f"docs/adr/ 至少需要 {MIN_ADR_COUNT} 篇 ADR，当前 {len(adr_files)} 篇")
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
    detail = f"README/ARCHITECTURE 存在，ADR {len(adr_files)} 篇且章节完整"
    return CheckResult("docs", True, detail, duration)


def check_cli(*, verbose: bool = False) -> CheckResult:
    started = time.perf_counter()
    python = project_python()
    problems: list[str] = []
    passed: list[str] = []
    outputs: list[str] = []
    for script_name in ENTRY_SCRIPTS:
        script = ROOT / script_name
        if not script.is_file():
            problems.append(f"缺少入口脚本 {script_name}")
            continue
        command = [python, script_name, "--help"]
        try:
            proc = run_command(command, timeout_s=CLI_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            problems.append(f"{script_name} --help 超时（>{CLI_TIMEOUT_S:.0f}s）")
            continue
        combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        if verbose:
            print(f"  命令: {' '.join(map(str, command))}")
        if proc.returncode != 0:
            problems.append(f"{script_name} --help 退出码 {proc.returncode}")
            outputs.append(f"--- {script_name} --help ---\n{_tail(combined)}")
        elif "usage" not in combined.lower():
            problems.append(f"{script_name} --help 输出中没有 usage 信息")
            outputs.append(f"--- {script_name} --help ---\n{_tail(combined)}")
        else:
            passed.append(script_name)
    duration = time.perf_counter() - started
    if problems:
        return CheckResult("cli", False, "；".join(problems), duration, "\n".join(outputs))
    return CheckResult("cli", True, f"{', '.join(passed)} --help 均可运行", duration)


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
