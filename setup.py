from setuptools import setup

requires = ["torch", "pytorch-nlp", "requests", "transpyle", "nltk", "bashlex"]
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
    packages=["nl2prog", "nl2prog.dataset", "nl2prog.dataset.django",
              "nl2prog.dataset.hearthstone", "nl2prog.dataset.nl2bash",
              "nl2prog.encoders", "nl2prog.language", "nl2prog.language.bash",
              "nl2prog.language.python", "nl2prog.metrics",
              "nl2prog.metrics.python", "nl2prog.nn", "nl2prog.nn.functional",
              "nl2prog.nn.nl2code", "nl2prog.nn.treegen", "nl2prog.nn.utils",
              "nl2prog.utils", "nl2prog.utils.data", "nl2prog.utils.transform",
              "nl2prog.utils.nl2code", "nl2prog.utils.python",
              "nl2prog.utils.treegen"])
