#! /bin/bash

flake8 $(find $(dirname $0)/../mlprogram -name "*.py") \
       $(find $(dirname $0)/../test -name "*.py")

black .

isort .
