#! /bin/bash
set -u

PROJECT_ROOT=$(dirname $0)/../../

python $PROJECT_ROOT/main.py src.interactive \
    --result $PROJECT_ROOT/result/django \
    $@
