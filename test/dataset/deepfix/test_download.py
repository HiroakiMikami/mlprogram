import gzip
import logging
import os
import sqlite3
import sys
import tempfile
import zipfile
from shutil import copyfile, copyfileobj

from mlprogram import Environment
from mlprogram.datasets.deepfix import download

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


class TestDownload(object):
    def test_download(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.pt")
            sqlitefile = os.path.join(tmpdir, "dataset.db")
            conn = sqlite3.connect(sqlitefile)
            c = conn.cursor()
            c.execute(
                "CREATE TABLE Code(code text, error text, errorcount int)")
            c.execute(
                "INSERT INTO Code VALUES ('foo', 'bar', 1)"
            )
            c.execute(
                "INSERT INTO Code VALUES ('foo', '', 0)"
            )
            conn.commit()
            conn.close()

            gzipfile = os.path.join(tmpdir, "dataset.gz")
            with gzip.open(gzipfile, "wb") as file, \
                    open(sqlitefile, "rb") as src_file:
                copyfileobj(src_file, file)
            path = os.path.join(tmpdir, "dataset.zip")
            with zipfile.ZipFile(path, "w") as z:
                with z.open(os.path.join("prutor-deepfix-09-12-2017",
                                         "prutor-deepfix-09-12-2017.db.gz"),
                            "w") as dst_file, \
                        open(gzipfile, "rb") as src_file:
                    copyfileobj(src_file, dst_file)

            def get(src, dst):
                copyfile(src, dst)
            dataset0 = download(cache_path=cache_path, path=path, get=get)

            def get2(src, dst):
                raise NotImplementedError
            dataset1 = download(cache_path=cache_path, path=path, get=get2)

        assert 2 == len(dataset0)
        assert dataset0[0] == Environment({"code": "foo", "error": "bar", "n_error": 1},
                                          set(["error", "n_error"]))
        assert dataset0[1] == Environment({"code": "foo", "error": "", "n_error": 0},
                                          set(["error", "n_error"]))

        assert list(dataset0) == list(dataset1)
