NL2Code-reimplementation
===

This repository is a reimplementation of NL2Code proposed in the [paper](https://arxiv.org/abs/1704.01696). NL2Code can synthesize the source code of a general purpose language (such as Python) from the natural language description.
The official implementation is available in the [GitHub repository](https://github.com/pcyin/NL2code/).


Notice (Dec/7/2019)
---

I rewrote the implementation from scratch. The previous one is in the [v0.0.0 tag](https://github.com/HiroakiMikami/NL2Code-reimplementation/tree/v0.0.0).


Performance
---

The details of the training results are shown in `result/` directory.
I showed the difference between the original repository and this one later. One of these differences may be the reason for the performance difference.

### *Django* Dataset

|         |Accuracy|BLEU4|
|---------|-------:|----:|
|original |   71.6%|84.5%|
|this repo|   42.8%|69.5%|

### Hearthstone Dataset

|         |Accuracy|BLEU4|
|---------|-------:|----:|
|original |   16.2%|75.8%|
|this repo|    3.0%|66.4%|

### nl2bash Dataset

|         |Accuracy|BLEU4|
|---------|-------:|----:|
|this repo|   15.2%|64.5%|


Difference between Official Implementation
---

I noticed that there are some differences between this implementation and the official one.
The followings summarize the differences.

#### 1. version of Python
The original implementation used Python 2.x, but this repository uses Python 3.x (I tested with the Python 3.7.4).

#### 2. tested dataset
Although the paper tested NL2Code with IFTTT dataset, this repository did not test with IFTTT dataset.

#### 3 definitions of action sequence
The original implementation added the cast action and did not support actions that have variadic children. This implementation omits the cast action and supports actions with variadic children.

#### 4. implementation of Dropout for LSTM
Dropout used in this repository and the original one is different. I tested dropout of the original repository, but it caused significant performance degradation (about x2 slower). So I decided to use more simple dropout implementations.

#### 5. maximum length of query and action sequences
The original implementation limits the length of the query and action sequence because Theano employes Define-and-Run style.
This implementation does not set the maximum length because of PyTorch Define-by-Run style.

Usage
---

### Requirements
#### Colab
* Google acount


#### Local Runtime
* Linux (I think the code works in macOSes, but I have not tested.)
* Python 3.x (tested with Python 3.7.4)


### Installation (if use local runtimes)

```bash
$ git clone https://github.com/HiroakiMikami/NL2Code-reimplementation nl2code
$ cd nl2code
$ pip install .
$ pip install . -e ["examples"]
$ pip install . -e ["colab"]
```


### Training with *Django* Dataset

*Warning The notebook in the examples directory will use Google Drive as data storage. Please be careful not to overwrite your data!*

The [*Django* dataset](https://github.com/odashi/ase15-django-dataset) is a collection of code with manually annotated descriptions. The code comes from the Django web framework.

The notebooks in `nl2code_examples/django` directory show how to train NL2Code with *Django* dataset. Training consists of the following steps:

1. Download dataset (`nl2code_examples/django/download_dataset.ipynb`)
2. Train and validate with the *Django* dataset (`nl2code_examples/django/train.ipynb`)


References
---

* [A Syntactic Neural Model for General-Purpose Code Generation, ACL2017](https://arxiv.org/abs/1704.01696)
* [the official GitHub repository](https://github.com/pcyin/NL2code/)
* [Learning to Generate Pseudo-code from Source Code Using Statistical Machine Translation, ACE2-15](https://ieeexplore.ieee.org/document/7372045)
* [Latent Predictor Networks for Code Generation](https://arxiv.org/abs/1603.06744)
* [nl2bash](https://github.com/TellinaTool/nl2bash)
