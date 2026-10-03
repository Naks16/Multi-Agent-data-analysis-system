"""Sandboxed executor for LLM-generated analysis code.

IMPORTANT: this is a *basic safeguard*, not a production-grade sandbox.
It stops obvious mistakes and naive misuse (importing os, calling eval, writing
files in random places), but a determined attacker can escape a Python-level
check. A real deployment would run generated code inside a locked-down
container (no network, read-only filesystem, CPU/memory limits), e.g. Docker
or gVisor.

Two layers of protection:
1. Static AST check (`check_code`) rejects dangerous imports/calls BEFORE running.
2. The code runs in a separate subprocess with a hard timeout, a temporary
   working directory, and a stripped-down environment (no API keys).
"""

import ast
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

DEFAULT_TIMEOUT_S = 30

# Modules the generated code may not import. The first six are from the spec;
# the rest are other common ways to reach the filesystem / processes / network.
FORBIDDEN_MODULES = {
    "os", "subprocess", "socket", "requests", "shutil", "sys",
    "pathlib", "importlib", "ctypes", "multiprocessing", "urllib", "http", "builtins",
}

# Built-in functions the generated code may not call.
FORBIDDEN_CALLS = {"eval", "exec", "__import__", "compile"}

# Only these environment variables are passed to the subprocess. Using an
# allowlist (instead of deleting known secrets) means API keys and any other
# secrets in the parent environment never reach generated code.
# SYSTEMROOT/PATH/TEMP are needed for Python and numpy to start on Windows.
ENV_ALLOWLIST = ("PATH", "SYSTEMROOT", "TEMP", "TMP", "LANG")


@dataclass
class ExecResult:
    success: bool
    stdout: str
    stderr: str
    error: str | None = None          # short human-readable reason for failure
    duration_s: float = 0.0
    timed_out: bool = False
    new_files: list[str] = field(default_factory=list)  # files created in output_dir


# --------------------------------------------------------------------------- #
# 1. Static check
# --------------------------------------------------------------------------- #

def _is_write_mode(call: ast.Call) -> bool:
    """Return True if an open(...) call may write. Unknown modes count as writing."""
    mode_node = call.args[1] if len(call.args) >= 2 else None
    for kw in call.keywords:
        if kw.arg == "mode":
            mode_node = kw.value
    if mode_node is None:
        return False  # default mode is "r"
    if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
        return any(ch in mode_node.value for ch in "wax+")
    return True  # mode is computed at runtime -> can't verify, assume write


def _path_is_inside_output_dir(path_node: ast.AST, output_dir: Path) -> bool:
    """Accept only paths we can verify statically:
    - a string literal that resolves inside output_dir, or
    - an f-string that starts with {OUTPUT_DIR} and contains no "..".
    """
    if isinstance(path_node, ast.Constant) and isinstance(path_node.value, str):
        target = Path(path_node.value)
        if not target.is_absolute():
            return False  # relative paths land in the temp cwd, not the output dir
        return target.resolve().is_relative_to(output_dir.resolve())

    if isinstance(path_node, ast.JoinedStr) and path_node.values:
        first = path_node.values[0]
        starts_with_output_dir = (
            isinstance(first, ast.FormattedValue)
            and isinstance(first.value, ast.Name)
            and first.value.id == "OUTPUT_DIR"
        )
        literal_parts = [v.value for v in path_node.values if isinstance(v, ast.Constant)]
        return starts_with_output_dir and not any(".." in p for p in literal_parts)

    return False


def check_code(code: str, output_dir: str | Path) -> list[str]:
    """Return a list of policy violations (empty list = code is allowed)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]

    output_dir = Path(output_dir)
    problems = []

    for node in ast.walk(tree):
        # import os / import os.path
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_MODULES:
                    problems.append(f"line {node.lineno}: import of '{alias.name}' is not allowed")

        # from os import path / from os.path import join
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_MODULES:
                problems.append(f"line {node.lineno}: import from '{node.module}' is not allowed")

        # Dunder attribute access (e.g. ().__class__.__subclasses__()) is the
        # classic way to escape checks like this one, so block it outright.
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            problems.append(f"line {node.lineno}: access to '{node.attr}' is not allowed")

        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in FORBIDDEN_CALLS:
                problems.append(f"line {node.lineno}: call to '{name}()' is not allowed")
            elif name == "open" and _is_write_mode(node):
                path_node = node.args[0] if node.args else None
                if path_node is None or not _path_is_inside_output_dir(path_node, output_dir):
                    problems.append(
                        f"line {node.lineno}: open() for writing is only allowed inside OUTPUT_DIR"
                    )

    return problems


# --------------------------------------------------------------------------- #
# 2. Subprocess execution
# --------------------------------------------------------------------------- #

def build_env() -> dict[str, str]:
    """Minimal environment for the child process (no API keys or other secrets)."""
    env = {k: os.environ[k] for k in ENV_ALLOWLIST if k in os.environ}
    env["MPLBACKEND"] = "Agg"                          # no GUI windows
    env["MPLCONFIGDIR"] = matplotlib.get_configdir()   # reuse font cache (fast startup)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _build_script(code: str, csv_path: Path, output_dir: Path) -> str:
    """Prepend the variables the generated code is allowed to use."""
    prelude = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        f"CSV_PATH = {str(csv_path)!r}\n"
        f"OUTPUT_DIR = {str(output_dir)!r}\n"
        "# ---- generated code below ----\n"
    )
    return prelude + code


def run_code(
    code: str,
    csv_path: str | Path,
    output_dir: str | Path,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ExecResult:
    """Check and run `code` in a subprocess. Never raises for bad code."""
    csv_path = Path(csv_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    problems = check_code(code, output_dir)
    if problems:
        return ExecResult(
            success=False, stdout="", stderr="",
            error="Code rejected by safety check:\n" + "\n".join(problems),
        )

    files_before = set(output_dir.iterdir())
    start = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="analyst_exec_") as workdir:
        script_path = Path(workdir) / "step.py"
        script_path.write_text(_build_script(code, csv_path, output_dir), encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=workdir,
                env=build_env(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            # subprocess.run kills the child before raising TimeoutExpired.
            return ExecResult(
                success=False,
                stdout=_to_text(e.stdout),
                stderr=_to_text(e.stderr),
                error=f"Timed out after {timeout_s}s",
                duration_s=time.perf_counter() - start,
                timed_out=True,
            )

    duration = time.perf_counter() - start
    new_files = sorted(str(p) for p in set(output_dir.iterdir()) - files_before)

    if proc.returncode != 0:
        # The last line of a traceback is usually the most useful summary.
        last_line = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
        return ExecResult(
            success=False, stdout=proc.stdout, stderr=proc.stderr,
            error=f"Exit code {proc.returncode}: {last_line}",
            duration_s=duration, new_files=new_files,
        )

    return ExecResult(
        success=True, stdout=proc.stdout, stderr=proc.stderr,
        duration_s=duration, new_files=new_files,
    )


def _to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
