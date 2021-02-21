import sys

from setuptools import find_packages, setup

requires = [
    "torch==1.7.1",
    "pytorch-nlp==0.5.0",
    "requests",
    "transpyle",
    "nltk",
    "bashlex",
    "pytorch-pfn-extras",
    "tqdm",
    "pycparser",
    "python-config @ git+https://github.com/HiroakiMikami/python-config",
]

if sys.version_info.major == 3 and sys.version_info.minor < 7:
    requires.append("dataclasses")

extras = {
    "test": [
        "flake8",
        "autopep8",
        "black",
        "isort",
        "mypy",
        "timeout-decorator",
        "pytest",
        "fairseq @ git+https://github.com/pytorch/fairseq",
    ],
}

setup(
    name="mlprogram",
    version="0.2.0",
    install_requires=requires,
    test_requires=extras["test"],
    extras_require=extras,
    packages=find_packages(),
)
