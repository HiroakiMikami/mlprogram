import os
from setuptools import setup, find_packages

requires = ["torch", "pytorch-nlp", "requests", "transpyle", "nltk", "bashlex",
            "docopt"]
extras = {
    "test": ["flake8", "autopep8", "fairseq"],
    "colab": ["jupyter", "jupyter-http-over-ws"],
    "visualize": ["graphviz"]
}

setup(
    name="nl2prog",
    install_requires=requires,
    test_requires=extras["test"],
    colab_requires=extras["colab"],
    visualize_requires=extras["visualize"],
    extras_require=extras,
    packages=find_packages() + [os.path.join("nl2prog", "dataset", "deepcoder",
                                             "DeepCoder_Utils")]
)
