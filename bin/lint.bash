#! /bin/bash

set -u
check=${1:-false}
flake8 $(find $(dirname $0)/../mlprogram -name "*.py") \
       $(find $(dirname $0)/../test -name "*.py") || exit 1

if [ "$check" = "false" ]
then
       black .
       isort .
else
       black . --diff || exit 1
       isort . --diff || exit 1
fi

