from setuptools import setup, find_packages
import sys

requires = ["torch", "pytorch-nlp", "requests", "transpyle", "nltk", "bashlex",
            "pytorch-pfn-extras", "tqdm", "pyyaml", "pycparser"]

if sys.version_info.major == 3 and sys.version_info.minor < 7:
    requires.append("dataclasses")

extras = {
    "test": ["flake8", "autopep8", "fairseq", "mypy==0.770",
             "timeout-decorator", "pytest", "pytest-parallel"],
    "colab": ["jupyter", "jupyter-http-over-ws"],
    "visualize": ["graphviz"]
}

setup(
    name="mlprogram",
    version="0.1.0",
    install_requires=requires,
    test_requires=extras["test"],
    colab_requires=extras["colab"],
    visualize_requires=extras["visualize"],
    extras_require=extras,
    packages=find_packages())
