"""Shared pytest fixtures for the Python test suite.

The gRPC stubs that ``prompt_app.app`` imports are gitignored; we
generate them and put them on ``sys.path`` at conftest-import time so
that test modules can ``from prompt_app.app import ...`` at the top.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_DIR = REPO_ROOT / "prompt_app" / "generated"


def _generate_stubs() -> None:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    (GEN_DIR / "__init__.py").touch()
    proto_dir = REPO_ROOT / "proto"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"--proto_path={proto_dir}",
            f"--python_out={GEN_DIR}",
            f"--grpc_python_out={GEN_DIR}",
            str(proto_dir / "prompt_app.proto"),
        ],
        check=True,
    )


if not (GEN_DIR / "prompt_app_pb2.py").exists():
    _generate_stubs()

for path in (str(REPO_ROOT), str(GEN_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)
