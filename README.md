NL2Prog
===

This repository is a library for synthesizing programs from natural language queries.
It provides 1) language-independent data structure for AST, 2) implementations of some NL-to-program DNNs, and 3) utilities for three commonly used datasets.


Notice (Dec/7/2019)
---

I rewrote the implementation from scratch. The previous one is in the [v0.0.0 tag](https://github.com/HiroakiMikami/NL2Prog/tree/v0.0.0).


Implemented Papers
---

### NL2Code

`nl2code.ipynb` in `examples` directory shows the usage of [NL2Code](https://arxiv.org/abs/1704.01696).

I noticed that there are some differences between this implementation and the official one.
The followings summarize the differences.

#### 1. version of Python
The original implementation used Python 2.x, but this repository uses Python 3.x (I tested with the Python 3.7.4).

#### 2. definitions of action sequence
The original implementation added the cast action and did not support actions that have variadic children. This implementation omits the cast action and supports actions with variadic children.

#### 3. implementation of Dropout for LSTM
Dropout used in this repository and the original one is different. I tested dropout of the original repository, but it caused significant performance degradation (about x2 slower). So I decided to use more simple dropout implementations.

#### 4. maximum length of query and action sequences
The original implementation limits the length of the query and action sequence because Theano employes Define-and-Run style.
This implementation does not set the maximum length because of PyTorch Define-by-Run style.


### TreeGen

Under implementation


Performance
---

The details of the training results are shown in `result/` directory.

### Django

|Paper  |Top-1 Acc|Top-1 BLEU|
|:---   |---:     |---:      |
|NL2Code|41.3%    |83.7%     |

### Hearthstone

|Paper  |Top-1 Acc|Top-1 BLEU|
|:---   |---:     |---:      |
|NL2Code|3.0%     |74.0%     |

### NL2Bash

|Paper   |Top-1 Acc|Top-1 BLEU|Top-3 Acc|Top-3 BLEU|
|:---    |---:     |---:      |---:     |---:      |
|NL2Code |6.1%     |55.6%     |12.3%    |63.3%     |


References
---

* [A Syntactic Neural Model for General-Purpose Code Generation, ACL2017](https://arxiv.org/abs/1704.01696)
* [TreeGen: A Tree-Based Transformer Architecture for Code Generation](https://arxiv.org/abs/1911.09983)
* [the official GitHub repository of NL2Code](https://github.com/pcyin/NL2code/)
* [the official GitHub repository of TreeGen](https://github.com/zysszy/TreeGen)
* [Learning to Generate Pseudo-code from Source Code Using Statistical Machine Translation, ACE2-15](https://ieeexplore.ieee.org/document/7372045)
* [Latent Predictor Networks for Code Generation](https://arxiv.org/abs/1603.06744)
* [nl2bash](https://github.com/TellinaTool/nl2bash)
