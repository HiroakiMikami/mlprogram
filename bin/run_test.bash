#! /bin/bash

python -m unittest discover $(dirname $0)/../test/language
python -m unittest discover $(dirname $0)/../test/language/python
python -m unittest discover $(dirname $0)/../test/nn
python -m unittest discover $(dirname $0)/../test/nn/utils
