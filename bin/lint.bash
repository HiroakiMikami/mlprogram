#! /bin/bash

flake8 $(find $(dirname $0)/../nl2prog -type d -name "DeepCoder_Utils" -prune -name "*.py") \
       $(find $(dirname $0)/../test -name "*.py") \
       $(find $(dirname $0)/../examples -name "*.py")
