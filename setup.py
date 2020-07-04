from setuptools import setup, find_packages
import sys

requires = ["torch", "pytorch-nlp", "requests", "transpyle", "nltk", "bashlex",
            "pytorch-pfn-extras", "tqdm"]

if sys.version_info.major == 3 and sys.version_info.minor < 7:
    requires.append("dataclasses")

extras = {
    "test": ["flake8", "autopep8", "fairseq", "mypy"],
    "colab": ["jupyter", "jupyter-http-over-ws"],
    "visualize": ["graphviz"]
}

setup(
    name="mlprogram",
    install_requires=requires,
    test_requires=extras["test"],
    colab_requires=extras["colab"],
    visualize_requires=extras["visualize"],
    extras_require=extras,
    packages=find_packages())
