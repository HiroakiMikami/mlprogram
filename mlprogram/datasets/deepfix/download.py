import gzip
import os
import sqlite3
import tempfile
import urllib
import zipfile
from shutil import copyfileobj
from typing import Callable

from mlprogram import Environment, logging
from mlprogram.datasets import DEFAULT_CACHE_DIR
from mlprogram.functools import file_cache
from mlprogram.utils.data import ListDataset

logger = logging.Logger(__name__)

BASE_PATH = "https://www.cse.iitk.ac.in/users/karkare/prutor/prutor-deepfix-09-12-2017.zip"  # noqa


def default_get(src: str, dst: str):
    with urllib.request.urlopen(src) as src_file, \
            open(dst, "wb") as dst_file:
        copyfileobj(src_file, dst_file)


def download(cache_path: str = os.path.join(DEFAULT_CACHE_DIR, "deepfix.pt"),
             path: str = BASE_PATH,
             get: Callable[[str, str], None] = default_get) \
        -> ListDataset:

    @file_cache(cache_path)
    def _download():
        logger.info("Download DeepFix dataset")
        logger.debug(f"Dataset path: {path}")
        samples = []
        with tempfile.TemporaryDirectory() as tmpdir:
            dst = os.path.join(tmpdir, "dataset.zip")
            get(path, dst)

            gzipfile = os.path.join(tmpdir, "dataset.gz")
            with zipfile.ZipFile(dst) as z:
                with z.open(os.path.join("prutor-deepfix-09-12-2017",
                                         "prutor-deepfix-09-12-2017.db.gz"),
                            "r") as file, \
                        open(gzipfile, "wb") as dst_file:
                    copyfileobj(file, dst_file)
            sqlitefile = os.path.join(tmpdir, "dataset.db")
            with gzip.open(gzipfile, "rb") as src_file, \
                    open(sqlitefile, "wb") as dst_file:
                copyfileobj(src_file, dst_file)

            conn = sqlite3.connect(sqlitefile)
            c = conn.cursor()
            for code, error, errorcount in \
                    c.execute("SELECT code, error, errorcount FROM Code"):
                samples.append(Environment(
                    inputs={"code": code},
                    supervisions={
                        "error": error,
                        "n_error": errorcount,
                    }
                ))
        return samples

    samples = _download()
    return ListDataset(samples)
