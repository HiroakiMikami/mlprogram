import os
import tempfile

import pytest

from mlprogram.functools import file_cache, with_file_cache


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_decorator(tmpdir):
    @file_cache(os.path.join(tmpdir, "tmp.pt"))
    def f():
        return len(os.listdir(tmpdir))

    assert f() == 0
    assert os.path.exists(os.path.join(tmpdir, "tmp.pt"))
    assert len(os.listdir(tmpdir)) == 1
    assert f() == 0


def test_with_file_cache(tmpdir):
    def f(d):
        return len(os.listdir(d))
    assert with_file_cache(os.path.join(tmpdir, "tmp.pt"), f, d=tmpdir) == 0
    assert os.path.exists(os.path.join(tmpdir, "tmp.pt"))
    assert len(os.listdir(tmpdir)) == 1
    assert with_file_cache(os.path.join(tmpdir, "tmp.pt"), f, d=tmpdir) == 0
