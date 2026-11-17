"""Import the MLCommons reference implementation (prompt, schema, scorer) as-is.

The reference lives in third_party/mlcommons-inference at the commit recorded in
docs/stage0.md. We import its modules directly instead of copying code, so the
prompt, the JSON schema, and the scoring functions are the official ones.

The reference package imports `mlperf_loadgen` at module import time, but the
parts we use (prompt construction, `ProductMetadata`, the F1 functions) never
call it. If LoadGen is not installed (it has no Windows wheel) an empty stand-in
module is registered so the import succeeds.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import types
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_REPO = ROOT / "third_party" / "mlcommons-inference"
REFERENCE_SRC = REFERENCE_REPO / "multimodal" / "qwen3-vl" / "src"
# Commit the project was developed against; score.py records the actual one.
EXPECTED_COMMIT = "3fbc329939999c13d0a7b5e67fb2092287e06047"


def _ensure_importable() -> bool:
    """Put the reference on sys.path. Returns True if LoadGen had to be stubbed."""
    if not REFERENCE_SRC.exists():
        raise FileNotFoundError(
            f"{REFERENCE_SRC} not found. Clone mlcommons/inference at {EXPECTED_COMMIT} "
            "into third_party/mlcommons-inference (see docs/stage0.md)."
        )
    if str(REFERENCE_SRC) not in sys.path:
        sys.path.insert(0, str(REFERENCE_SRC))
    try:
        importlib.import_module("mlperf_loadgen")
    except ImportError:
        sys.modules["mlperf_loadgen"] = types.ModuleType("mlperf_loadgen")
        return True
    return False


@lru_cache(maxsize=1)
def load() -> types.SimpleNamespace:
    """Return the official modules: `.evaluation`, `.schema`, `.task`, plus metadata."""
    stubbed = _ensure_importable()
    return types.SimpleNamespace(
        evaluation=importlib.import_module("mlperf_inf_mm_q3vl.evaluation"),
        schema=importlib.import_module("mlperf_inf_mm_q3vl.schema"),
        task=importlib.import_module("mlperf_inf_mm_q3vl.task"),
        loadgen_stubbed=stubbed,
        commit=reference_commit(),
    )


def reference_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REFERENCE_REPO), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
