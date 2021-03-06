mlprogram
===

A Library of Deep Learning (Machine Learning) for Programming Tasks.
It provides a toolbox for implementing and evaluating deep learning methods related to programming.

Purpose
---

The main purpose of this repository is making my experiments easy. Recently, many papers proposed deep learning methods for programming, such as programming by example and auto reparing. But many of them requires complex and error-prone implementations. For example, beam search decoding with a programming language grammar is complex and unique to this task. I want to create and maintain well-tested implementations of such algorithms.

### Focuses

* Library for handling programming languages in deep learning tasks
* Utilities for benchmark datasets of various tasks
* Simple baseline solution for program generation

Now I do not place value on re-implementing exsiting papers.
The machine learning for programming field is still immature. There are no de-fact benchmark tasks in this field (such as image classification w/ ImageNet and object detection w/ COCO in the image). Also, there are no de-fact model (such as ResNet in the image). 


Feature Lists and Plans
---

* Benchmark dataset
    * Auto Reparing
        * DeepFix: [the official repository](https://bitbucket.org/iiscseal/deepfix/src/master/)
    * Program Synthesis from Natural Language
        * Hearthstone: [Latent Predictor Networks for Code Generation](https://arxiv.org/abs/1603.06744)
        * Django: [Learning to Generate Pseudo-code from Source Code Using Statistical Machine Translation, ACE2-15](https://ieeexplore.ieee.org/document/7372045)
        * NL2Bash: [nl2bash](https://github.com/TellinaTool/nl2bash)
        * (TODO) Spider: [Spider 1.0 Yale Semantic Parsing and Text-to-SQL Challenge](https://yale-lily.github.io/spider)
    * Programming by Examples
        * 2D CSG
        * (TODO) DeepCoder
        * (TODO) ShapeNet
* Deep Learning Models
    * Attention Based LSTM
    * AST LSTM (based on NL2Code)
* ProgramSynthesis Methods
    * supervised training
    * reinforcement learning for programming by example
    * (TODO) interpreter arppoximated by DNN
* Other Papers
    * [NL2Code](https://arxiv.org/abs/1704.01696): [the official repository](https://github.com/pcyin/NL2code/)
    * [TreeGen](https://arxiv.org/abs/1911.09983): [the official repository](https://github.com/zysszy/TreeGen)
    * [PbE with REPL](http://arxiv.org/abs/1906.04604): [the official repository](https://github.com/flxsosa/ProgramSearch)


Benchmark
---

### NL2Prog (Hearthstone)

|Method|#params [MiB]|training time [min]|max time per example [sec]|BLEU@top1|config name|
:-----|-------------:|------------------:|-------------------------:|--------:|:----------|
|tree LSTM|       7.7|                 92|                        15|  0.75020|`hearthstone/baseline_evaluate_short`|
|tree LSTM|       7.7|                 92|                       180|  0.76540|`hearthstone/baseline_evaluate_long`|


### Programming by Example without Inputs (CSG)

|Method                          |#params [MiB]|training time [min]|max time per example [sec]|generation rate|config file|
|:-------------------------------|------------:|------------------:|-------------------------:|---------------:|:----------|
|tree LSTM                       |16           |75                 |30                        |18/30|`csg/baseline_evaluate_short`|
|tree LSTM                       |16           |75                 |360                       |22/30|`csg/baseline_evaluate_long`|
|tree LSTM + REINFORCESynthesizer|16           |75                 |30                        |18/30|`csg/baseline_evaluate_rl_synthesizer_short`|
|tree LSTM + REINFORCESynthesizer|16           |75                 |360                       |22/30|`csg/baseline_evaluate_rl_synthesizer_short`|


### Auto Repair

TODO

Usage Examples
---

`tools/launch.py` is the launcher script and `configs` directory contains the examples.

### Train/Evaluate NL2Code with Hearthstone Dataset

It requires CUDA enabled GPU.

```bash
$ python tools/launch.py --config configs/nl2code/nl2code_train.py
$ python tools/launch.py --config configs/nl2code/nl2code_evaluate.py
```


Warning
---

* The implementation is highly experimental, and I may change it significantly.
* The reproduced algorithms may be different from the authors' implementations. For example, the original implementation of NL2Code uses the grammar of Python 2.7.x while this repository uses the grammar of running Python version.



<!--
Implemented Papers
---

### NL2Code

[NL2Code](https://arxiv.org/abs/1704.01696) creates programs (ASTs) from natural language descriptions. It encodes the structure of ASTs.
There is no big known issue.

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

[TreeGen](https://arxiv.org/abs/1911.09983) creates programs (ASTs) from natural language descriptions. It uses the Transformer as a DNN model.
There are the two known issues:

* The loss sometimes becomes NaN in the final phase of the training
* The result when using Hearthstone dataset is worse than the reported value.


### Write, Execute, Assess: Program Synthesis with a REPL

[PbE with REPL](http://arxiv.org/abs/1906.04604) creates programs (ASTs) from input/output examples. It uses reinforcement learning for better generalization.
There are many known issues in the implementation. The most crucial one is that RL training is very unstable. I cannot improve model performance by using RL.
-->
