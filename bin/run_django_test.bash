#! /bin/bash

python -m unittest discover $(dirname $0)/../examples/django/test
