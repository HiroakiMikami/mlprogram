import os
import re
import subprocess
import tempfile
from typing import IO, List, cast

from mlprogram.languages import Analyzer as BaseAnalyzer


class Analyzer(BaseAnalyzer[str, str]):
    def __init__(self, clang_cmd: str = "clang"):
        self.clang_cmd = clang_cmd

    def __call__(self, code: str) -> List[str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "x.o")
            proc = subprocess.Popen(
                [self.clang_cmd, "-x", "c", "-c", "-", "-o", path],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True)
            cast(IO[str], proc.stdin).write(code)
            _, stderr = proc.communicate()
        errors = []
        text = ""
        for line in stderr.split("\n"):
            if re.match(r'<stdin>:\d+:\d+:\s+error:', line):
                # error
                if text != "":
                    errors.append(text)
                text = line
            elif re.match(r'<stdin>:\d+:\d+:\s+warning:', line):
                # warning
                if text != "":
                    errors.append(text)
                text = line
            elif re.match(r'\d+ \w+ generated.', line):
                # summary of the result
                continue
            elif re.match(r'\s*', line):
                continue
            else:
                text = text + "\n" + line
        if text != "":
            errors.append(text)

        return errors
