#! /bin/bash
set -u

PROJECT_ROOT=$(dirname $0)/../

python -m unittest discover $PROJECT_ROOT/test/nn/layers
python -m unittest discover $PROJECT_ROOT/test/nn
python -m unittest discover $PROJECT_ROOT/test/python
python -m unittest discover $PROJECT_ROOT/test
