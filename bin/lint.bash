#! /bin/bash

flake8 $(find $(dirname $0)/../nl2code -name "*.py") \
       $(find $(dirname $0)/../test -name "*.py") \
       $(find $(dirname $0)/../nl2code_examples -name "*.py")
