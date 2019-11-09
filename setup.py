from setuptools import setup

requires = ["torch", "pytorch-nlp"]
extras = {
    "test": ["flake8", "autopep8"],
    "django": ["transpyle", "nltk", "tensorboard"],
    "visualize": ["graphviz"]
}

setup(
    name="nl2code",
    install_requires=requires,
    test_requires=extras["test"],
    django_requires=extras["django"],
    visualize_requires=extras["visualize"],
    extras_require=extras)
