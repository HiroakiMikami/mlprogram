import os
import tempfile

from mlprogram.functools import file_cache


class TestFileCache(object):
    def test_decorator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            @file_cache(os.path.join(tmpdir, "tmp.pt"))
            def f():
                return len(os.listdir(tmpdir))

            assert f() == 0
            assert os.path.exists(os.path.join(tmpdir, "tmp.pt"))
            assert len(os.listdir(tmpdir)) == 1
            assert f() == 0
