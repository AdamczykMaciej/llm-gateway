"""The shipped types let a consumer's pyright run infer structured-output results."""

import subprocess
import sys
from pathlib import Path

import pytest

CONSUMER = Path(__file__).parent / "typing" / "consumer.py"


def test_consumer_example_type_checks():
    pyright = Path(sys.executable).parent / "pyright"
    if not pyright.exists():
        pytest.skip("pyright is not installed")
    proc = subprocess.run(
        [str(pyright), "--pythonpath", sys.executable, str(CONSUMER)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
