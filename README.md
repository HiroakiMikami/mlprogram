mlprogram
===

This repository is the library of deep learning techniques for programming.
It provides 1) implementation to handle grammars of programming languages, 2) utilities for commonly used datasets, and 3) reimplementations of some algorithms.

Notice (Aug/30/2020)
---

I rewrote the implementation significantly. I uses [pytorch-pfn-extras](https://github.com/pfnet/pytorch-pfn-extras) instead of [gin-config](https://github.com/google/gin-config) as a config system.


Purpose
---

This repository aims at providing the utilities of deep learning methods for programming. 
Many papers have recently proposed deep learning methods for programming, such as programming by examples and auto repairing. But many methods require complex and error-prone algorithms. For example, beam search decoding with a programming language grammar is complex and unique to this task. I want to make my experiments easy by creating well-tested implementations of such algorithms.


Warning
---

The implementation is highly experimental, and I may change it significantly.

The reproduced algorithms may be different from the authors' implementations. For example, the original implementation of NL2Code uses the grammar of Python 2.7.x while this repository uses the grammar of running Python version.


Implemented Papers
---

### NL2Code

[NL2Code](https://arxiv.org/abs/1704.01696) creates programs (ASTs) from natural language descriptions. It encodes the structure of ASTs 
There is no big known issue.

<!--
#### 1. version of Python
The original implementation used Python 2.x, but this repository uses Python 3.x (I tested with the Python 3.7.4).

#### 2. definitions of action sequence
The original implementation added the cast action and did not support actions that have variadic children. This implementation omits the cast action and supports actions with variadic children.

#### 3. implementation of Dropout for LSTM
Dropout used in this repository and the original one is different. I tested dropout of the original repository, but it caused significant performance degradation (about x2 slower). So I decided to use more simple dropout implementations.

#### 4. maximum length of query and action sequences
The original implementation limits the length of the query and action sequence because Theano employes Define-and-Run style.
This implementation does not set the maximum length because of PyTorch Define-by-Run style.
-->


### TreeGen

[TreeGen](https://arxiv.org/abs/1911.09983) creates programs (ASTs) from natural language descriptions. It uses the Transformer as a DNN model.
There are the two known issues:

* The loss sometimes becomes NaN in the final phase of the training
* The result when using Hearthstone dataset is worse than the reported value.


### Write, Execute, Assess: Program Synthesis with a REPL

[PbE with REPL](http://arxiv.org/abs/1906.04604) creates programs (ASTs) from input/output examples. It uses reinforcement learning for better generalization.
There are many known issues in the implementation. The most crucial one is that RL training is very unstable. I cannot improve model performance by using RL.


Usage Examples
---

`tools/launch.py` is the launcher script and `configs` directory contains the examples.

### Train/Evaluate NL2Code with Hearthstone Dataset

It requires CUDA enabled GPU.

```bash
$ python tools/launch.py --config configs/nl2code/nl2code_train.py
$ python tools/launch.py --config configs/nl2code/nl2code_evaluate.py
```


Reference
---

* [the official GitHub repository of NL2Code](https://github.com/pcyin/NL2code/)
* [the official GitHub repository of TreeGen](https://github.com/zysszy/TreeGen)
* [the official GitHub repository of PbE with REPL](https://github.com/flxsosa/ProgramSearch)
* [Learning to Generate Pseudo-code from Source Code Using Statistical Machine Translation, ACE2-15](https://ieeexplore.ieee.org/document/7372045)
* [Latent Predictor Networks for Code Generation](https://arxiv.org/abs/1603.06744)
* [nl2bash](https://github.com/TellinaTool/nl2bash)
