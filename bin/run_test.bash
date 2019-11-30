#! /bin/bash

python -m unittest discover $(dirname $0)/../test/language
python -m unittest discover $(dirname $0)/../test/language/python
python -m unittest discover $(dirname $0)/../test/nn
python -m unittest discover $(dirname $0)/../test/nn/utils
python -m unittest discover $(dirname $0)/../test/

python -m unittest discover $(dirname $0)/../nl2code_examples/django/test
python -m unittest discover $(dirname $0)/../nl2code_examples/hearthstone/test
