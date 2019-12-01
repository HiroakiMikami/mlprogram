from setuptools import setup

requires = ["torch", "pytorch-nlp"]
extras = {
    "test": ["flake8", "autopep8"],
    "examples": ["transpyle", "nltk", "bashlex", "tensorboard"],
    "colab": ["jupyter", "jupyter-http-over-ws"],
    "visualize": ["graphviz"]
}

setup(
    name="nl2code",
    install_requires=requires,
    test_requires=extras["test"],
    examples_requires=extras["examples"],
    colab_requires=extras["colab"],
    visualize_requires=extras["visualize"],
    extras_require=extras)
