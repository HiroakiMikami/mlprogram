import io
import os
import tempfile
from contextlib import redirect_stderr

import pytest

from mlprogram.launch.launch import launch


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_launch(tmpdir):
    run = os.path.join(tmpdir, "file.py")
    with open(run, "w") as f:
        f.write("""
import sys
def main():
    print("executed", file=sys.stderr)
""")

    stderr = io.StringIO()
    with redirect_stderr(stderr):
        launch(run, None, tmpdir, 0, 1, [])
    assert stderr.getvalue() == "executed\n"


def test_launch_with_options(tmpdir):
    run = os.path.join(tmpdir, "file.py")
    with open(run, "w") as f:
        f.write("""
import sys
from mlprogram.launch import global_options
global_options["arg"] = "value"
def main():
    print(global_options.arg, file=sys.stderr)
""")
    stderr = io.StringIO()
    with redirect_stderr(stderr):
        launch(run, None, tmpdir, 0, 1, ["--arg", "v"])
    assert stderr.getvalue() == "v\n"
