import os
import tempfile
import zipfile
import gzip
import sqlite3
import logging
import sys
from shutil import copyfileobj, copyfile
from mlprogram import Environment
from mlprogram.datasets.deepfix import download

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


class TestDownload(object):
    def test_download(self):
        with tempfile.TemporaryDirectory() as tmpdir:
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
            dataset = download(path=path, get=get)

        assert 2 == len(dataset)
        assert dataset[0] == Environment(
            inputs={"code": "foo"},
            supervisions={"error": "bar", "n_error": 1})
        assert dataset[1] == Environment(
            inputs={"code": "foo"},
            supervisions={"error": "", "n_error": 0})
