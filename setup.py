import sys

from setuptools import find_packages, setup

requires = [
    "torch==1.7.0",
    "pytorch-nlp==0.5.0",
    "requests",
    "transpyle",
    "nltk",
    "bashlex",
    "pytorch-pfn-extras==0.3.0",
    "tqdm",
    "pyyaml",
    "pycparser",
]

if sys.version_info.major == 3 and sys.version_info.minor < 7:
    requires.append("dataclasses")

extras = {
    "test": [
        "flake8",
        "autopep8",
        "black",
        "isort",
        "mypy==0.770",
        "timeout-decorator",
        "pytest",
        "pytest-parallel",
        "fairseq",
    ],
    "colab": ["jupyter", "jupyter-http-over-ws"],
    "visualize": ["graphviz"],
}

setup(
    name="mlprogram",
    version="0.1.0",
    install_requires=requires,
    test_requires=extras["test"],
    colab_requires=extras["colab"],
    visualize_requires=extras["visualize"],
    extras_require=extras,
    packages=find_packages(),
)
